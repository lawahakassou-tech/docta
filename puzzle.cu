// puzzle.cu - Recherche de clés privées Bitcoin (secp256k1) multi-cibles : TOUS les GPU (CUDA) + TOUS les cœurs CPU.
//
// Version hybride pour Kaggle / Google Colab / tout Linux avec CUDA. Sans dépendance externe (hors CUDA runtime).
// Les GPU et les cœurs CPU travaillent ensemble sur la même suite de blocs : chaque bloc n'est testé qu'une fois.
//
// Compilation (voir readme.txt pour le détail et les cellules prêtes à l'emploi) :
//   nvcc -O3 -std=c++17 -arch=native -Xcompiler -O3,-march=native,-pthread -lpthread -o puzzle puzzle.cu
//   (Si -arch=native n'est pas reconnu : -gencode arch=compute_75,code=sm_75 pour T4, compute_60 pour P100,
//    compute_80 pour A100, compute_89 pour L4, compute_90 pour H100 ...)
//   Version CPU seule, sans CUDA (tests, machine sans GPU) : g++ -x c++ -O3 -march=native -std=c++17 -pthread -o puzzle_cpu puzzle.cu
//
// Usage :
//   puzzle <clé_publique | adresse | cibles.txt | cibles.bloom> <bits | début:fin> [address] [mode] [options]
//   puzzle --info                                   liste les GPU et les cœurs détectés
//   puzzle --selftest                               lance uniquement les auto-tests
//   puzzle --genpub <clé_privée_hex> | --genaddr <clé_privée_hex>     (utilitaires de test)
//
// La clé recherchée est dans [2^(bits-1) , 2^bits - 1], ou dans la plage début:fin donnée en hexadécimal.
// address      : à ajouter quand les cibles sont des adresses Bitcoin (1... ou bc1q...)
// cibles.txt   : une cible par ligne -> construit et sauvegarde cibles.bloom ; cibles.bloom : chargement direct
// Mode (un seul par lancement) :   -r aléatoire (défaut)   -s séquentiel   -sobol   -pi   -e   -ln   -m Monte Carlo
// -v R         : voisinage : chaque clé générée par la suite est le CENTRE d'une fenêtre [k-R, k+R] testée en entier.
// Matériel :
//   -t N          nombre de cœurs CPU utilisés (défaut : tous ; 0 = GPU seuls)
//   --gpus L      GPU utilisés : all (défaut) | none | liste d'indices, ex. 0,1
//   --no-gpu      équivaut à --gpus none
// Sorties / reprise :
//   --out DIR     dossier des résultats et du point de reprise (ex. /kaggle/working ou /content/drive/MyDrive/puzzle)
//   --ckpt FILE   fichier de reprise (défaut : DIR/puzzle_<empreinte>.ckpt) ; --no-ckpt le désactive ; --fresh l'ignore
//   --ckpt-every S  intervalle de sauvegarde en secondes (défaut 120)
//   --time T      arrêt propre après T (ex. 3600, 90m, 11h30m) : le point de reprise est écrit
//   --status S    intervalle (s) des lignes d'état hors terminal (défaut 15)
//   --skip-test   saute les auto-tests au démarrage (utile pour une reprise rapide)
#define _FILE_OFFSET_BITS 64
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <csignal>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <atomic>
#include <exception>
#include <ctime>
#include <thread>
#include <filesystem>
#include <unistd.h>
#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

typedef uint64_t u64;
typedef uint32_t u32;
typedef unsigned __int128 u128;

// Vrai CUDA (compilé par nvcc) ou émulation hôte (g++ seul / auto-tests sans GPU)
#ifdef __CUDACC__
  #define REAL_CUDA 1
#else
  #define REAL_CUDA 0
#endif
#ifdef __CUDACC__
  #define HD __host__ __device__
#else
  #define HD
#endif


struct U256 { u64 d[4]; };               // little-endian (d[0] = poids faible)
static const U256 ZERO = {{0,0,0,0}};
static const U256 ONE  = {{1,0,0,0}};
static const U256 PRIME = {{0xFFFFFFFEFFFFFC2FULL, ~0ULL, ~0ULL, ~0ULL}};
static const u64  KC = 0x1000003D1ULL;   // 2^256 mod p

// ---------------------------------------------------------------- 256 bits brut
static inline bool is_zero(const U256& a){ return (a.d[0]|a.d[1]|a.d[2]|a.d[3])==0; }
static inline bool eq(const U256& a,const U256& b){
    return ((a.d[0]^b.d[0])|(a.d[1]^b.d[1])|(a.d[2]^b.d[2])|(a.d[3]^b.d[3]))==0;
}
static inline bool geq(const U256& a,const U256& b){
    for(int i=3;i>=0;--i){ if(a.d[i]!=b.d[i]) return a.d[i]>b.d[i]; }
    return true;
}
static inline u64 add_raw(U256& r,const U256& a,const U256& b){
    u128 acc=0; u64 c=0;
    for(int i=0;i<4;i++){ acc=(u128)a.d[i]+b.d[i]+c; r.d[i]=(u64)acc; c=(u64)(acc>>64); }
    return c;
}
static inline u64 sub_raw(U256& r,const U256& a,const U256& b){
    u64 br=0;
    for(int i=0;i<4;i++){
        u128 t=(u128)a.d[i]-b.d[i]-br;
        r.d[i]=(u64)t; br=(u64)((t>>64)&1);
    }
    return br;
}
static U256 div_small(U256 a,u64 d){                       // a / d (d > 0)
    u128 rem=0;
    for(int i=3;i>=0;--i){ u128 cur=(rem<<64)|a.d[i]; a.d[i]=(u64)(cur/d); rem=cur%d; }
    return a;
}
static U256 shr_n(U256 a,int n){
    for(int s2=0;s2<n;s2++){ for(int k=0;k<4;k++){ a.d[k]>>=1; if(k<3) a.d[k]|=a.d[k+1]<<63; } }
    return a;
}
static inline void add_small(U256& a,u64 v){
    u128 acc=(u128)a.d[0]+v; a.d[0]=(u64)acc; u64 c=(u64)(acc>>64);
    for(int i=1;i<4;i++){ acc=(u128)a.d[i]+c; a.d[i]=(u64)acc; c=(u64)(acc>>64); }
}
static inline int get_bit(const U256& a,int i){ return (int)((a.d[i>>6]>>(i&63))&1); }
static int highest_bit(const U256& a){
    for(int i=255;i>=0;--i) if(get_bit(a,i)) return i;
    return -1;
}
static U256 shr2(U256 a){
    for(int i=0;i<4;i++){
        a.d[i]>>=2;
        if(i<3) a.d[i]|=a.d[i+1]<<62;   // a.d[i+1] n'a pas encore été décalé
    }
    return a;
}

// ---------------------------------------------------------------- hex / décimal
static bool parse_hex(const std::string& s,U256& out){
    if(s.empty()||s.size()>64) return false;
    out=ZERO;
    for(size_t i=0;i<s.size();i++){
        char c=s[s.size()-1-i]; int v;
        if(c>='0'&&c<='9') v=c-'0'; else if(c>='a'&&c<='f') v=c-'a'+10;
        else if(c>='A'&&c<='F') v=c-'A'+10; else return false;
        out.d[i/16]|=(u64)v<<((i%16)*4);
    }
    return true;
}
static std::string to_hex(const U256& a){
    char buf[65];
    snprintf(buf,sizeof buf,"%016llx%016llx%016llx%016llx",
        (unsigned long long)a.d[3],(unsigned long long)a.d[2],
        (unsigned long long)a.d[1],(unsigned long long)a.d[0]);
    return buf;
}
static std::string to_dec(U256 a){
    if(is_zero(a)) return "0";
    std::vector<u64> parts;
    while(!is_zero(a)){
        u128 rem=0;
        for(int i=3;i>=0;--i){
            u128 cur=(rem<<64)|a.d[i];
            a.d[i]=(u64)(cur/1000000000ULL);
            rem=cur%1000000000ULL;
        }
        parts.push_back((u64)rem);
    }
    char buf[32]; std::string s;
    snprintf(buf,sizeof buf,"%llu",(unsigned long long)parts.back()); s+=buf;
    for(int i=(int)parts.size()-2;i>=0;--i){ snprintf(buf,sizeof buf,"%09llu",(unsigned long long)parts[i]); s+=buf; }
    return s;
}

// ---------------------------------------------------------------- corps Fp (p = 2^256 - 2^32 - 977)
static inline void fe_add(U256& r,const U256& a,const U256& b){
    U256 t; u64 c=add_raw(t,a,b);
    if(c){ U256 k={{KC,0,0,0}}; add_raw(t,t,k); }
    if(geq(t,PRIME)) sub_raw(t,t,PRIME);
    r=t;
}
static inline void fe_sub(U256& r,const U256& a,const U256& b){
    U256 t; u64 br=sub_raw(t,a,b);
    if(br){ U256 k={{KC,0,0,0}}; sub_raw(t,t,k); }
    r=t;
}
// Réduction d'un produit de 512 bits modulo p (2^256 = 2^32 + 977 mod p)
static inline void fe_reduce(U256& r,const u64* t){
    u64 r5[5]; u64 carry=0;
    for(int i=0;i<4;i++){
        u128 acc=(u128)t[4+i]*KC+t[i]+carry;
        r5[i]=(u64)acc; carry=(u64)(acc>>64);
    }
    r5[4]=carry;
    U256 res; u128 acc=(u128)r5[4]*KC+r5[0];
    res.d[0]=(u64)acc; carry=(u64)(acc>>64);
    for(int i=1;i<4;i++){ acc=(u128)r5[i]+carry; res.d[i]=(u64)acc; carry=(u64)(acc>>64); }
    if(carry){
        acc=(u128)res.d[0]+KC; res.d[0]=(u64)acc; carry=(u64)(acc>>64);
        for(int i=1;i<4;i++){ acc=(u128)res.d[i]+carry; res.d[i]=(u64)acc; carry=(u64)(acc>>64); }
    }
    if(geq(res,PRIME)) sub_raw(res,res,PRIME);
    r=res;
}
static inline void fe_mul(U256& r,const U256& a,const U256& b){
    u64 t[8]={0,0,0,0,0,0,0,0};
    for(int i=0;i<4;i++){
        u64 carry=0;
        for(int j=0;j<4;j++){
            u128 acc=(u128)a.d[i]*b.d[j]+t[i+j]+carry;
            t[i+j]=(u64)acc; carry=(u64)(acc>>64);
        }
        t[i+4]=carry;
    }
    fe_reduce(r,t);
}
// Carré dédié : 10 produits de 64 bits au lieu de 16
static inline void fe_sqr(U256& r,const U256& a){
    u64 t[8]={0,0,0,0,0,0,0,0};
    for(int i=0;i<3;i++){
        u64 carry=0;
        for(int j=i+1;j<4;j++){
            u128 acc=(u128)a.d[i]*a.d[j]+t[i+j]+carry;
            t[i+j]=(u64)acc; carry=(u64)(acc>>64);
        }
        t[i+4]=carry;
    }
    u64 c=0;
    for(int k=0;k<8;k++){ u64 v=t[k]; t[k]=(v<<1)|c; c=v>>63; }
    u64 carry=0;
    for(int i=0;i<4;i++){
        u128 acc=(u128)a.d[i]*a.d[i]+t[2*i]+carry;
        t[2*i]=(u64)acc; carry=(u64)(acc>>64);
        acc=(u128)t[2*i+1]+carry;
        t[2*i+1]=(u64)acc; carry=(u64)(acc>>64);
    }
    fe_reduce(r,t);
}
static inline void fe_sqr_n(U256& r,const U256& a,int n){
    U256 t=a; for(int i=0;i<n;i++) fe_sqr(t,t); r=t;
}
static U256 fe_pow(const U256& base,const U256& e){
    U256 r=ONE; int top=highest_bit(e);
    for(int i=top;i>=0;--i){ fe_sqr(r,r); if(get_bit(e,i)) fe_mul(r,r,base); }
    return r;
}
// Inverse modulaire a^(p-2) par chaîne d'additions : 255 carrés + 15 produits
static U256 fe_inv(const U256& a){
    U256 x2,x3,x6,x9,x11,x22,x44,x88,x176,x220,x223,t;
    fe_sqr(x2,a); fe_mul(x2,x2,a);
    fe_sqr(x3,x2); fe_mul(x3,x3,a);
    fe_sqr_n(x6,x3,3); fe_mul(x6,x6,x3);
    fe_sqr_n(x9,x6,3); fe_mul(x9,x9,x3);
    fe_sqr_n(x11,x9,2); fe_mul(x11,x11,x2);
    fe_sqr_n(x22,x11,11); fe_mul(x22,x22,x11);
    fe_sqr_n(x44,x22,22); fe_mul(x44,x44,x22);
    fe_sqr_n(x88,x44,44); fe_mul(x88,x88,x44);
    fe_sqr_n(x176,x88,88); fe_mul(x176,x176,x88);
    fe_sqr_n(x220,x176,44); fe_mul(x220,x220,x44);
    fe_sqr_n(x223,x220,3); fe_mul(x223,x223,x3);
    fe_sqr_n(t,x223,23); fe_mul(t,t,x22);
    fe_sqr_n(t,t,5); fe_mul(t,t,a);
    fe_sqr_n(t,t,3); fe_mul(t,t,x2);
    fe_sqr_n(t,t,2); fe_mul(t,t,a);
    return t;
}

// ---------------------------------------------------------------- courbe y^2 = x^3 + 7
struct Aff { U256 x,y; bool inf; };
struct Jac { U256 X,Y,Z; bool inf; };

static Aff GEN;
static const char* GX_HEX="79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798";
static const char* GY_HEX="483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8";

static bool on_curve(const Aff& P){
    U256 l,r; fe_sqr(l,P.y); fe_sqr(r,P.x); fe_mul(r,r,P.x);
    U256 seven={{7,0,0,0}}; fe_add(r,r,seven);
    return eq(l,r);
}

// --- Arithmétique affine (lente, 1 inversion par opération) : sert à la vérification indépendante
static Aff aff_add(const Aff& P,const Aff& Q){
    if(P.inf) return Q;
    if(Q.inf) return P;
    U256 l;
    if(eq(P.x,Q.x)){
        if(!eq(P.y,Q.y) || is_zero(P.y)){ Aff o; o.x=ZERO;o.y=ZERO;o.inf=true; return o; }
        U256 n,d; fe_sqr(n,P.x); U256 n3; fe_add(n3,n,n); fe_add(n3,n3,n);   // 3x^2
        fe_add(d,P.y,P.y);
        fe_mul(l,n3,fe_inv(d));
    } else {
        U256 n,d; fe_sub(n,Q.y,P.y); fe_sub(d,Q.x,P.x);
        fe_mul(l,n,fe_inv(d));
    }
    Aff R; R.inf=false;
    fe_sqr(R.x,l); fe_sub(R.x,R.x,P.x); fe_sub(R.x,R.x,Q.x);
    U256 t; fe_sub(t,P.x,R.x); fe_mul(R.y,l,t); fe_sub(R.y,R.y,P.y);
    return R;
}
static Aff aff_mul(const U256& k,const Aff& P){
    Aff R; R.x=ZERO;R.y=ZERO;R.inf=true;
    for(int i=highest_bit(k);i>=0;--i){
        R=aff_add(R,R);
        if(get_bit(k,i)) R=aff_add(R,P);
    }
    return R;
}

// --- Arithmétique Jacobienne (rapide) : sert au calcul du point de départ
static Jac jac_dbl(const Jac& P){
    if(P.inf||is_zero(P.Y)){ Jac o; o.X=o.Y=o.Z=ZERO; o.inf=true; return o; }
    U256 A,B,C,D,E,F,t,t2;
    fe_sqr(A,P.X); fe_sqr(B,P.Y); fe_sqr(C,B);
    fe_add(t,P.X,B); fe_sqr(t,t); fe_sub(t,t,A); fe_sub(t,t,C); fe_add(D,t,t);
    fe_add(E,A,A); fe_add(E,E,A);
    fe_sqr(F,E);
    Jac R; R.inf=false;
    fe_add(t,D,D); fe_sub(R.X,F,t);
    fe_sub(t,D,R.X); fe_mul(t,E,t);
    fe_add(t2,C,C); fe_add(t2,t2,t2); fe_add(t2,t2,t2);   // 8C
    fe_sub(R.Y,t,t2);
    fe_mul(t,P.Y,P.Z); fe_add(R.Z,t,t);
    return R;
}
static Jac jac_madd(const Jac& P,const Aff& Q){
    if(P.inf){ Jac R; R.X=Q.x;R.Y=Q.y;R.Z=ONE;R.inf=false; return R; }
    U256 Z1Z1,U2,S2,H,HH,I,J,r,V,t;
    fe_sqr(Z1Z1,P.Z); fe_mul(U2,Q.x,Z1Z1);
    fe_mul(S2,Q.y,P.Z); fe_mul(S2,S2,Z1Z1);
    fe_sub(H,U2,P.X);
    fe_sub(t,S2,P.Y); fe_add(r,t,t);
    if(is_zero(H)){
        if(is_zero(r)) return jac_dbl(P);
        Jac o; o.X=o.Y=o.Z=ZERO; o.inf=true; return o;
    }
    fe_sqr(HH,H); fe_add(I,HH,HH); fe_add(I,I,I);
    fe_mul(J,H,I); fe_mul(V,P.X,I);
    Jac R; R.inf=false;
    fe_sqr(R.X,r); fe_sub(R.X,R.X,J); fe_sub(R.X,R.X,V); fe_sub(R.X,R.X,V);
    fe_sub(t,V,R.X); fe_mul(t,r,t);
    U256 u; fe_mul(u,P.Y,J); fe_add(u,u,u);
    fe_sub(R.Y,t,u);
    fe_add(t,P.Z,H); fe_sqr(t,t); fe_sub(t,t,Z1Z1); fe_sub(R.Z,t,HH);
    return R;
}
static Jac jac_mul(const U256& k,const Aff& P){
    Jac R; R.X=R.Y=R.Z=ZERO; R.inf=true;
    for(int i=highest_bit(k);i>=0;--i){
        R=jac_dbl(R);
        if(get_bit(k,i)) R=jac_madd(R,P);
    }
    return R;
}
static Aff jac_to_aff(const Jac& J){
    Aff A; A.x=ZERO;A.y=ZERO;
    if(J.inf){ A.inf=true; return A; }
    U256 zi=fe_inv(J.Z),zi2,zi3;
    fe_sqr(zi2,zi); fe_mul(zi3,zi2,zi);
    fe_mul(A.x,J.X,zi2); fe_mul(A.y,J.Y,zi3); A.inf=false;
    return A;
}

// ---------------------------------------------------------------- utilitaires divers
static std::string compress_hex(const Aff& P){
    return std::string(((P.y.d[0]&1)?"03":"02"))+to_hex(P.x);
}
static inline u64 splitmix(u64& s){
    u64 z=(s+=0x9E3779B97F4A7C15ULL);
    z=(z^(z>>30))*0xBF58476D1CE4E5B9ULL; z=(z^(z>>27))*0x94D049BB133111EBULL;
    return z^(z>>31);
}
struct Rng {
    u64 s[4];
    void seed(u64 v){ for(int i=0;i<4;i++) s[i]=splitmix(v); }
    static inline u64 rotl(u64 x,int k){ return (x<<k)|(x>>(64-k)); }
    u64 next(){
        u64 r=rotl(s[1]*5,7)*9, t=s[1]<<17;
        s[2]^=s[0]; s[3]^=s[1]; s[1]^=s[2]; s[0]^=s[3]; s[2]^=t; s[3]=rotl(s[3],45);
        return r;
    }
};
struct SpinLock {
    std::atomic_flag f=ATOMIC_FLAG_INIT;
    void lock(){ while(f.test_and_set(std::memory_order_acquire)){} }
    void unlock(){ f.clear(std::memory_order_release); }
};
static U256 shl_u64(u64 v,int sh){
    U256 r=ZERO; int limb=sh>>6, off=sh&63;
    if(limb<4) r.d[limb]|=v<<off;
    if(off && limb+1<4) r.d[limb+1]|=v>>(64-off);
    return r;
}
static U256 low_mask(int nbits){           // 2^nbits - 1
    U256 r=ZERO;
    for(int b=0;b<nbits;b++) r.d[b>>6]|=1ULL<<(b&63);
    return r;
}
static std::string fmt_int(u64 v){
    std::string s=std::to_string(v), o;
    int c=0;
    for(int i=(int)s.size()-1;i>=0;--i){ o.insert(o.begin(),s[i]); if(++c%3==0&&i>0) o.insert(o.begin(),' '); }
    return o;
}
static std::string fmt_keys(long double v){
    const char* u[]={"","K","M","G","T","P","E"};
    int i=0; while(v>=1000.0L && i<6){ v/=1000.0L; i++; }
    char b[32]; snprintf(b,sizeof b,i?"%.2Lf %s":"%.0Lf %s",v,u[i]); return b;
}
static std::string fmt_time(long double s){
    char b[64];
    if(s<0||s!=s) return "?";
    if(s>3.15e9L) return ">100 ans";
    unsigned long long t=(unsigned long long)s;
    unsigned long long y=t/31536000ULL; t%=31536000ULL;
    unsigned long long d=t/86400, h=(t%86400)/3600, m=(t%3600)/60, sec=t%60;
    if(y) snprintf(b,sizeof b,"%llu an(s) %llu j",y,d);
    else if(d) snprintf(b,sizeof b,"%llu j %02lluh %02llum",d,h,m);
    else snprintf(b,sizeof b,"%02llu:%02llu:%02llu",h,m,sec);
    return b;
}
static std::string short_hex(const U256& v){
    std::string h=to_hex(v); size_t p=h.find_first_not_of('0');
    h=(p==std::string::npos)?"0":h.substr(p);
    if(h.size()>22) h=h.substr(0,10)+"…"+h.substr(h.size()-10);
    return "0x"+h;
}

// ---------------------------------------------------------------- SHA-256 / RIPEMD-160 (un seul bloc) et adresses
static inline u32 rotr32(u32 x,int n){ return (x>>n)|(x<<(32-n)); }
static inline u32 rotl32(u32 x,int n){ return (x<<n)|(x>>(32-n)); }
static const u32 SHA_K[64]={
0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};

// SHA-256 d'un message tenant dans un seul bloc (w = 16 mots big-endian déjà préparés)
static inline void sha256_block(const u32* w16,u32* out){
    u32 w[64];
    for(int i=0;i<16;i++) w[i]=w16[i];
    #pragma GCC unroll 64
    for(int i=16;i<64;i++){
        u32 s0=rotr32(w[i-15],7)^rotr32(w[i-15],18)^(w[i-15]>>3);
        u32 s1=rotr32(w[i-2],17)^rotr32(w[i-2],19)^(w[i-2]>>10);
        w[i]=w[i-16]+s0+w[i-7]+s1;
    }
    u32 a=0x6a09e667,b=0xbb67ae85,c=0x3c6ef372,d=0xa54ff53a,e=0x510e527f,f=0x9b05688c,g=0x1f83d9ab,h=0x5be0cd19;
    #pragma GCC unroll 64
    for(int i=0;i<64;i++){
        u32 S1=rotr32(e,6)^rotr32(e,11)^rotr32(e,25), ch=(e&f)^(~e&g);
        u32 t1=h+S1+ch+SHA_K[i]+w[i];
        u32 S0=rotr32(a,2)^rotr32(a,13)^rotr32(a,22), mj=(a&b)^(a&c)^(b&c);
        u32 t2=S0+mj;
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    out[0]=a+0x6a09e667; out[1]=b+0xbb67ae85; out[2]=c+0x3c6ef372; out[3]=d+0xa54ff53a;
    out[4]=e+0x510e527f; out[5]=f+0x9b05688c; out[6]=g+0x1f83d9ab; out[7]=h+0x5be0cd19;
}
// SHA-256 d'un message de n <= 55 octets (octet par octet)
static void sha256_bytes(const unsigned char* m,size_t n,unsigned char* out32){
    unsigned char buf[64]; memset(buf,0,64); memcpy(buf,m,n); buf[n]=0x80;
    u64 bl=(u64)n*8; for(int i=0;i<8;i++) buf[63-i]=(unsigned char)(bl>>(8*i));
    u32 w[16],o[8];
    for(int i=0;i<16;i++) w[i]=((u32)buf[4*i]<<24)|((u32)buf[4*i+1]<<16)|((u32)buf[4*i+2]<<8)|buf[4*i+3];
    sha256_block(w,o);
    for(int i=0;i<8;i++){ out32[4*i]=o[i]>>24; out32[4*i+1]=o[i]>>16; out32[4*i+2]=o[i]>>8; out32[4*i+3]=o[i]; }
}

static const unsigned char RMD_R1[80]={0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15, 7,4,13,1,10,6,15,3,12,0,9,5,2,14,11,8,
    3,10,14,4,9,15,8,1,2,7,0,6,13,11,5,12, 1,9,11,10,0,8,12,4,13,3,7,15,14,5,6,2, 4,0,5,9,7,12,2,10,14,1,3,8,11,6,15,13};
static const unsigned char RMD_R2[80]={5,14,7,0,9,2,11,4,13,6,15,8,1,10,3,12, 6,11,3,7,0,13,5,10,14,15,8,12,4,9,1,2,
    15,5,1,3,7,14,6,9,11,8,12,2,10,0,4,13, 8,6,4,1,3,11,15,0,5,12,2,13,9,7,10,14, 12,15,10,4,1,5,8,7,6,2,13,14,0,3,9,11};
static const unsigned char RMD_S1[80]={11,14,15,12,5,8,7,9,11,13,14,15,6,7,9,8, 7,6,8,13,11,9,7,15,7,12,15,9,11,7,13,12,
    11,13,6,7,14,9,13,15,14,8,13,6,5,12,7,5, 11,12,14,15,14,15,9,8,9,14,5,6,8,6,5,12, 9,15,5,11,6,8,13,12,5,12,13,14,11,8,5,6};
static const unsigned char RMD_S2[80]={8,9,9,11,13,15,15,5,7,7,8,11,14,14,12,6, 9,13,15,7,12,8,9,11,7,7,12,7,6,15,13,11,
    9,7,15,11,8,6,6,14,12,13,5,14,13,13,7,5, 15,5,8,11,14,14,6,14,6,9,12,9,12,5,15,8, 8,5,12,9,12,5,14,6,8,13,6,5,15,13,11,11};
static const u32 RMD_K1[5]={0x00000000,0x5A827999,0x6ED9EBA1,0x8F1BBCDC,0xA953FD4E};
static const u32 RMD_K2[5]={0x50A28BE6,0x5C4DD124,0x6D703EF3,0x7A6D76E9,0x00000000};
template<int F> static inline u32 rmd_f(u32 x,u32 y,u32 z){
    if(F==0) return x^y^z;
    if(F==1) return (x&y)|(~x&z);
    if(F==2) return (x|~y)^z;
    if(F==3) return (x&z)|(y&~z);
    return x^(y|~z);
}
// RIPEMD-160 d'un message tenant dans un seul bloc (X = 16 mots little-endian)
template<int F> static inline void rmd_seg(const u32* X,u32& a,u32& b,u32& c,u32& d,u32& e,
                                           u32& a2,u32& b2,u32& c2,u32& d2,u32& e2){
    #pragma GCC unroll 16
    for(int j=16*F;j<16*F+16;j++){
        u32 t=rotl32(a+rmd_f<F>(b,c,d)+X[RMD_R1[j]]+RMD_K1[F],RMD_S1[j])+e;
        a=e; e=d; d=rotl32(c,10); c=b; b=t;
        t=rotl32(a2+rmd_f<4-F>(b2,c2,d2)+X[RMD_R2[j]]+RMD_K2[F],RMD_S2[j])+e2;
        a2=e2; e2=d2; d2=rotl32(c2,10); c2=b2; b2=t;
    }
}
static inline void rmd160_block(const u32* X,u32* out){
    u32 a=0x67452301,b=0xEFCDAB89,c=0x98BADCFE,d=0x10325476,e=0xC3D2E1F0;
    u32 a2=a,b2=b,c2=c,d2=d,e2=e;
    rmd_seg<0>(X,a,b,c,d,e,a2,b2,c2,d2,e2);
    rmd_seg<1>(X,a,b,c,d,e,a2,b2,c2,d2,e2);
    rmd_seg<2>(X,a,b,c,d,e,a2,b2,c2,d2,e2);
    rmd_seg<3>(X,a,b,c,d,e,a2,b2,c2,d2,e2);
    rmd_seg<4>(X,a,b,c,d,e,a2,b2,c2,d2,e2);
    const u32 I0=0x67452301u, I1=0xEFCDAB89u, I2=0x98BADCFEu, I3=0x10325476u, I4=0xC3D2E1F0u;
    u32 t=I1+c+d2;
    out[1]=I2+d+e2; out[2]=I3+e+a2; out[3]=I4+a+b2; out[4]=I0+b+c2; out[0]=t;
}
// hash160 empaqueté dans un U256 : lecture little-endian des 20 octets du condensat
static inline U256 pack_rmd(const u32* h){
    U256 r; r.d[0]=h[0]|((u64)h[1]<<32); r.d[1]=h[2]|((u64)h[3]<<32); r.d[2]=h[4]; r.d[3]=0; return r;
}
static U256 pack20(const unsigned char* b){
    U256 r=ZERO; memcpy(&r.d[0],b,8); memcpy(&r.d[1],b+8,8); u32 t; memcpy(&t,b+16,4); r.d[2]=t; return r;
}
static void unpack20(const U256& h,unsigned char* b){
    memcpy(b,&h.d[0],8); memcpy(b+8,&h.d[1],8); u32 t=(u32)h.d[2]; memcpy(b+16,&t,4);
}
// Chemin rapide (boucle de recherche) : hash160 de la clé publique compressée (parité de Y, X)
static inline U256 hash160_fast(const U256& x,u64 par){
    u32 X[8];
    X[0]=(u32)(x.d[3]>>32); X[1]=(u32)x.d[3]; X[2]=(u32)(x.d[2]>>32); X[3]=(u32)x.d[2];
    X[4]=(u32)(x.d[1]>>32); X[5]=(u32)x.d[1]; X[6]=(u32)(x.d[0]>>32); X[7]=(u32)x.d[0];
    u32 w[16],H[8],rw[16],o[5];
    w[0]=((0x02u+(u32)par)<<24)|(X[0]>>8);
    for(int i=1;i<8;i++) w[i]=(X[i-1]<<24)|(X[i]>>8);
    w[8]=(X[7]<<24)|0x00800000u;
    for(int i=9;i<15;i++) w[i]=0;
    w[15]=264;
    sha256_block(w,H);
    for(int i=0;i<8;i++) rw[i]=__builtin_bswap32(H[i]);
    rw[8]=0x80; for(int i=9;i<14;i++) rw[i]=0; rw[14]=256; rw[15]=0;
    rmd160_block(rw,o);
    return pack_rmd(o);
}
// Chemin "octets" (vérifications indépendantes)
static void pub33(const U256& x,u64 par,unsigned char* out){
    out[0]=(unsigned char)(2+par);
    for(int i=0;i<32;i++) out[1+i]=(unsigned char)(x.d[3-i/8]>>(56-8*(i%8)));
}
static void hash160_bytes(const unsigned char* m,size_t n,unsigned char* out20){
    unsigned char dg[32]; sha256_bytes(m,n,dg);
    u32 X[16]={0},o[5];
    for(int i=0;i<8;i++) X[i]=(u32)dg[4*i]|((u32)dg[4*i+1]<<8)|((u32)dg[4*i+2]<<16)|((u32)dg[4*i+3]<<24);
    X[8]=0x80; X[14]=256;
    rmd160_block(X,o);
    for(int i=0;i<5;i++){ out20[4*i]=o[i]; out20[4*i+1]=o[i]>>8; out20[4*i+2]=o[i]>>16; out20[4*i+3]=o[i]>>24; }
}
static U256 hash160_point_bytes(const Aff& P){
    unsigned char pb[33],h[20]; pub33(P.x,P.y.d[0]&1,pb); hash160_bytes(pb,33,h); return pack20(h);
}


// ---- hash160 de plusieurs clés à la fois (vecteurs SIMD via les extensions GCC/Clang ; version scalaire sous nvcc)
#if defined(__CUDACC__) || defined(NO_SIMD)
#define SCALAR_HASH 1
#endif
#ifndef VLANES
#define VLANES 8          // 256 bits (AVX2 si -march=native)
#endif
#ifdef SCALAR_HASH
static void hash160_vec(const U256* xs,const u64* pars,U256* out){
    for(int l=0;l<VLANES;l++) out[l]=hash160_fast(xs[l],pars[l]);
}
#else
typedef u32 vu __attribute__((vector_size(4*VLANES)));
static inline vu vsplat(u32 x){ vu v; for(int i=0;i<VLANES;i++) v[i]=x; return v; }
#define VROTR(x,n) (((x)>>(n))|((x)<<(32-(n))))
#define VROTL(x,n) (((x)<<(n))|((x)>>(32-(n))))
template<int F> static inline vu rmd_fv(vu x,vu y,vu z){
    if(F==0) return x^y^z;
    if(F==1) return (x&y)|(~x&z);
    if(F==2) return (x|~y)^z;
    if(F==3) return (x&z)|(y&~z);
    return x^(y|~z);
}
template<int F> static inline void rmd_segv(const vu* X,vu& a,vu& b,vu& c,vu& d,vu& e,vu& a2,vu& b2,vu& c2,vu& d2,vu& e2){
    #pragma GCC unroll 16
    for(int j=16*F;j<16*F+16;j++){
        vu t=VROTL(a+rmd_fv<F>(b,c,d)+X[RMD_R1[j]]+vsplat(RMD_K1[F]),RMD_S1[j])+e;
        a=e; e=d; d=VROTL(c,10); c=b; b=t;
        t=VROTL(a2+rmd_fv<4-F>(b2,c2,d2)+X[RMD_R2[j]]+vsplat(RMD_K2[F]),RMD_S2[j])+e2;
        a2=e2; e2=d2; d2=VROTL(c2,10); c2=b2; b2=t;
    }
}
// hash160 de VLANES clés publiques compressées (xs[l], pars[l]) -> out[l] (même empaquetage que hash160_fast)
static void hash160_vec(const U256* xs,const u64* pars,U256* out){
    u32 Xl[8][VLANES], Pl[VLANES];
    for(int l=0;l<VLANES;l++){
        const U256& x=xs[l];
        Xl[0][l]=(u32)(x.d[3]>>32); Xl[1][l]=(u32)x.d[3]; Xl[2][l]=(u32)(x.d[2]>>32); Xl[3][l]=(u32)x.d[2];
        Xl[4][l]=(u32)(x.d[1]>>32); Xl[5][l]=(u32)x.d[1]; Xl[6][l]=(u32)(x.d[0]>>32); Xl[7][l]=(u32)x.d[0];
        Pl[l]=0x02u+(u32)pars[l];
    }
    vu X[8],w[64],H[8],rw[16];
    for(int i=0;i<8;i++) memcpy(&X[i],Xl[i],sizeof(vu));
    vu P; memcpy(&P,Pl,sizeof(vu));
    w[0]=(P<<24)|(X[0]>>8);
    for(int i=1;i<8;i++) w[i]=(X[i-1]<<24)|(X[i]>>8);
    w[8]=(X[7]<<24)|vsplat(0x00800000u);
    for(int i=9;i<15;i++) w[i]=vsplat(0);
    w[15]=vsplat(264);
    #pragma GCC unroll 64
    for(int i=16;i<64;i++){
        vu s0=VROTR(w[i-15],7)^VROTR(w[i-15],18)^(w[i-15]>>3);
        vu s1=VROTR(w[i-2],17)^VROTR(w[i-2],19)^(w[i-2]>>10);
        w[i]=w[i-16]+s0+w[i-7]+s1;
    }
    vu a=vsplat(0x6a09e667u),b=vsplat(0xbb67ae85u),c=vsplat(0x3c6ef372u),d=vsplat(0xa54ff53au);
    vu e=vsplat(0x510e527fu),f=vsplat(0x9b05688cu),g=vsplat(0x1f83d9abu),h=vsplat(0x5be0cd19u);
    #pragma GCC unroll 64
    for(int i=0;i<64;i++){
        vu S1=VROTR(e,6)^VROTR(e,11)^VROTR(e,25), ch=(e&f)^(~e&g);
        vu t1=h+S1+ch+vsplat(SHA_K[i])+w[i];
        vu S0=VROTR(a,2)^VROTR(a,13)^VROTR(a,22), mj=(a&b)^(a&c)^(b&c);
        vu t2=S0+mj;
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    H[0]=a+vsplat(0x6a09e667u); H[1]=b+vsplat(0xbb67ae85u); H[2]=c+vsplat(0x3c6ef372u); H[3]=d+vsplat(0xa54ff53au);
    H[4]=e+vsplat(0x510e527fu); H[5]=f+vsplat(0x9b05688cu); H[6]=g+vsplat(0x1f83d9abu); H[7]=h+vsplat(0x5be0cd19u);
    for(int i=0;i<8;i++){
        vu q=H[i];
        rw[i]=(q>>24)|((q>>8)&vsplat(0x0000FF00u))|((q<<8)&vsplat(0x00FF0000u))|(q<<24);
    }
    rw[8]=vsplat(0x80); for(int i=9;i<14;i++) rw[i]=vsplat(0); rw[14]=vsplat(256); rw[15]=vsplat(0);
    vu a1=vsplat(0x67452301u),b1=vsplat(0xEFCDAB89u),c1=vsplat(0x98BADCFEu),d1=vsplat(0x10325476u),e1=vsplat(0xC3D2E1F0u);
    vu a2=a1,b2=b1,c2=c1,d2=d1,e2=e1;
    rmd_segv<0>(rw,a1,b1,c1,d1,e1,a2,b2,c2,d2,e2);
    rmd_segv<1>(rw,a1,b1,c1,d1,e1,a2,b2,c2,d2,e2);
    rmd_segv<2>(rw,a1,b1,c1,d1,e1,a2,b2,c2,d2,e2);
    rmd_segv<3>(rw,a1,b1,c1,d1,e1,a2,b2,c2,d2,e2);
    rmd_segv<4>(rw,a1,b1,c1,d1,e1,a2,b2,c2,d2,e2);
    vu o[5];
    o[0]=vsplat(0xEFCDAB89u)+c1+d2; o[1]=vsplat(0x98BADCFEu)+d1+e2; o[2]=vsplat(0x10325476u)+e1+a2;
    o[3]=vsplat(0xC3D2E1F0u)+a1+b2; o[4]=vsplat(0x67452301u)+b1+c2;
    u32 T[5][VLANES];
    for(int k=0;k<5;k++) memcpy(T[k],&o[k],sizeof(vu));
    for(int l=0;l<VLANES;l++){
        out[l].d[0]=T[0][l]|((u64)T[1][l]<<32); out[l].d[1]=T[2][l]|((u64)T[3][l]<<32);
        out[l].d[2]=T[4][l]; out[l].d[3]=0;
    }
}
#endif


// ---- Base58Check (P2PKH) et Bech32 (P2WPKH)
static const char* B58="123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
static std::string b58check_encode(unsigned char version,const unsigned char* h20){
    std::vector<unsigned char> d(25); d[0]=version; memcpy(&d[1],h20,20);
    unsigned char s1[32],s2[32]; sha256_bytes(d.data(),21,s1); sha256_bytes(s1,32,s2); memcpy(&d[21],s2,4);
    size_t zeros=0; while(zeros<d.size()&&d[zeros]==0) zeros++;
    std::vector<unsigned char> num(d.begin()+zeros,d.end()); std::string out;
    while(!num.empty()){
        std::vector<unsigned char> q; unsigned rem=0;
        for(unsigned char c:num){ unsigned cur=rem*256+c; unsigned qb=cur/58; rem=cur%58; if(!q.empty()||qb) q.push_back((unsigned char)qb); }
        out.push_back(B58[rem]); num=q;
    }
    out.append(zeros,'1'); std::reverse(out.begin(),out.end());
    return out;
}
// 0 = invalide, 1 = P2PKH valide (h20 rempli), 2 = valide mais non supporté (P2SH ...)
static int b58check_decode(const std::string& s,unsigned char* h20){
    if(s.empty()||s.size()>40) return 0;
    std::vector<unsigned char> num;                      // little-endian
    size_t zeros=0; while(zeros<s.size()&&s[zeros]=='1') zeros++;
    for(char ch:s){
        const char* p=strchr(B58,ch); if(!p||!ch) return 0;
        unsigned carry=(unsigned)(p-B58);
        for(size_t j=0;j<num.size();j++){ carry+=(unsigned)num[j]*58; num[j]=(unsigned char)carry; carry>>=8; }
        while(carry){ num.push_back((unsigned char)carry); carry>>=8; }
    }
    std::vector<unsigned char> d(zeros,0); d.insert(d.end(),num.rbegin(),num.rend());
    if(d.size()!=25) return 0;
    unsigned char s1[32],s2[32]; sha256_bytes(d.data(),21,s1); sha256_bytes(s1,32,s2);
    if(memcmp(s2,&d[21],4)!=0) return 0;
    if(d[0]!=0x00) return 2;
    memcpy(h20,&d[1],20); return 1;
}
static const char* B32="qpzry9x8gf2tvdw0s3jn54khce6mua7l";
static u32 bech32_polymod(const std::vector<unsigned char>& v){
    static const u32 G[5]={0x3b6a57b2,0x26508e6d,0x1ea119fa,0x3d4233dd,0x2a1462b3};
    u32 chk=1;
    for(unsigned char x:v){ u32 b=chk>>25; chk=((chk&0x1ffffff)<<5)^x; for(int i=0;i<5;i++) if((b>>i)&1) chk^=G[i]; }
    return chk;
}
static std::vector<unsigned char> bech32_hrp_expand(const std::string& hrp){
    std::vector<unsigned char> v; for(char c:hrp) v.push_back((unsigned char)c>>5);
    v.push_back(0); for(char c:hrp) v.push_back((unsigned char)c&31); return v;
}
static std::string bech32_encode_p2wpkh(const unsigned char* h20){
    std::vector<unsigned char> data; data.push_back(0);
    int acc=0,bits=0;
    for(int i=0;i<20;i++){ acc=(acc<<8)|h20[i]; bits+=8; while(bits>=5){ bits-=5; data.push_back((acc>>bits)&31); } }
    if(bits) data.push_back((acc<<(5-bits))&31);
    std::vector<unsigned char> v=bech32_hrp_expand("bc"); v.insert(v.end(),data.begin(),data.end());
    for(int i=0;i<6;i++) v.push_back(0);
    u32 pm=bech32_polymod(v)^1;
    for(int i=0;i<6;i++) data.push_back((pm>>(5*(5-i)))&31);
    std::string out="bc1"; for(unsigned char x:data) out.push_back(B32[x]);
    return out;
}
static int bech32_decode_addr(const std::string& s0,unsigned char* h20){
    if(s0.size()<14||s0.size()>90) return 0;
    bool lo=false,up=false; for(char c:s0){ if(c>='a'&&c<='z') lo=true; if(c>='A'&&c<='Z') up=true; }
    if(lo&&up) return 0;
    std::string s=s0; for(auto& c:s) if(c>='A'&&c<='Z') c+=32;
    size_t pos=s.rfind('1'); if(pos==std::string::npos||pos<1||s.size()-pos<7) return 0;
    std::string hrp=s.substr(0,pos); std::vector<unsigned char> data;
    for(size_t i=pos+1;i<s.size();i++){ const char* p=strchr(B32,s[i]); if(!p||!s[i]) return 0; data.push_back((unsigned char)(p-B32)); }
    std::vector<unsigned char> v=bech32_hrp_expand(hrp); v.insert(v.end(),data.begin(),data.end());
    u32 pm=bech32_polymod(v);
    if(pm!=1&&pm!=0x2bc830a3) return 0;
    if(hrp!="bc") return 0;
    data.resize(data.size()-6);
    if(data.empty()) return 0;
    if(data[0]!=0||pm!=1) return 2;                      // taproot (v1+) ... : non supporté
    std::vector<unsigned char> prog; int acc=0,bits=0;
    for(size_t i=1;i<data.size();i++){ acc=(acc<<5)|data[i]; bits+=5; if(bits>=8){ bits-=8; prog.push_back((acc>>bits)&255); } }
    if(bits>=5||((acc<<(8-bits))&255)) return 0;
    if(prog.size()!=20) return 2;                        // P2WSH (32 octets) : non supporté
    memcpy(h20,prog.data(),20); return 1;
}
// ---------------------------------------------------------------- ensemble de cibles (Bloom + liste exacte)
struct Ent { U256 x; u64 par; };                       // X de la clé publique + parité de Y (0 = préfixe 02, 1 = 03)
static_assert(sizeof(Ent)==40,"taille Ent");
static inline bool x_less(const U256& a,const U256& b){
    for(int i=3;i>=0;--i){ if(a.d[i]!=b.d[i]) return a.d[i]<b.d[i]; }
    return false;
}
static bool ent_less(const Ent& a,const Ent& b){
    if(!eq(a.x,b.x)) return x_less(a.x,b.x);
    return a.par<b.par;
}
struct TargetSet {
    std::vector<Ent> list;                              // triée, sans doublon
    std::vector<u64> bloom;                             // nblocks * 8 mots de 64 bits (blocs de 512 bits)
    u64 nblocks=0;
    u64 src_hash=0;
    bool addr=false;                                    // true : la liste contient des hash160 (adresses)
};
// Filtre de Bloom "par blocs" : 1 bloc (une ligne de cache) par test, 6 bits par clé.
// X est uniforme : on lit directement ses bits, sans fonction de hachage.
static inline bool bloom_test(const u64* bl,u64 mask,const U256& x){
    const u64* blk=bl+((x.d[0]&mask)<<3);
    u64 h=x.d[1];
    for(int j=0;j<6;j++){
        u64 p=(h>>(9*j))&511;
        if(!((blk[p>>6]>>(p&63))&1)) return false;
    }
    return true;
}
static inline void bloom_insert(u64* bl,u64 mask,const U256& x){
    u64* blk=bl+((x.d[0]&mask)<<3);
    u64 h=x.d[1];
    for(int j=0;j<6;j++){ u64 p=(h>>(9*j))&511; blk[p>>6]|=1ULL<<(p&63); }
}
static void build_bloom(TargetSet& ts){
    u64 n=ts.list.size(), need=(n+15)/16; if(need<1) need=1;
    u64 nb=1; while(nb<need) nb<<=1;
    ts.nblocks=nb; ts.bloom.assign(nb*8,0);
    for(const Ent& e:ts.list) bloom_insert(ts.bloom.data(),nb-1,e.x);
}
static void finalize_targets(TargetSet& ts){
    std::sort(ts.list.begin(),ts.list.end(),ent_less);
    ts.list.erase(std::unique(ts.list.begin(),ts.list.end(),
        [](const Ent& a,const Ent& b){ return eq(a.x,b.x)&&a.par==b.par; }),ts.list.end());
    build_bloom(ts);
}
static bool ts_has_x(const TargetSet& ts,const U256& x){
    Ent probe; probe.x=x; probe.par=0;
    auto it=std::lower_bound(ts.list.begin(),ts.list.end(),probe,ent_less);
    return it!=ts.list.end() && eq(it->x,x);
}
static bool ts_has_exact(const TargetSet& ts,const U256& x,u64 par){
    Ent probe; probe.x=x; probe.par=par;
    return std::binary_search(ts.list.begin(),ts.list.end(),probe,ent_less);
}
static std::string ent_hex(const Ent& e){ return std::string(e.par?"03":"02")+to_hex(e.x); }
static std::string ent_hash_hex(const Ent& e){
    unsigned char b[20]; unpack20(e.x,b); char s[41]; for(int i=0;i<20;i++) snprintf(s+2*i,3,"%02x",b[i]); return s;
}
static std::string ent_addr_legacy(const Ent& e){ unsigned char b[20]; unpack20(e.x,b); return b58check_encode(0x00,b); }
static std::string ent_addr_bech32(const Ent& e){ unsigned char b[20]; unpack20(e.x,b); return bech32_encode_p2wpkh(b); }
static std::string ent_text(const TargetSet& ts,const Ent& e){ return ts.addr? ent_addr_legacy(e) : ent_hex(e); }
// Adresse -> Ent (hash160 empaqueté). Retour : 0 invalide, 1 ok, 2 valide mais non supportée (P2SH, P2WSH, Taproot)
static int parse_address(const std::string& s,Ent& e){
    unsigned char h[20];
    int r=b58check_decode(s,h);
    if(r==0) r=bech32_decode_addr(s,h);
    if(r==1){ e.x=pack20(h); e.par=0; }
    return r;
}

static bool parse_pub(const std::string& s,Ent& e){
    if(s.size()!=66||s[0]!='0'||(s[1]!='2'&&s[1]!='3')) return false;
    U256 x; if(!parse_hex(s.substr(2),x)||geq(x,PRIME)) return false;
    static U256 ex; static bool init=false;
    if(!init){ ex=PRIME; add_small(ex,1); ex=shr2(ex); init=true; }
    U256 r,y,y2; fe_sqr(r,x); fe_mul(r,r,x); U256 seven={{7,0,0,0}}; fe_add(r,r,seven);
    y=fe_pow(r,ex); fe_sqr(y2,y);
    if(!eq(y2,r)) return false;                          // X n'est pas sur la courbe
    e.x=x; e.par=(s[1]=='3'); return true;
}
static void parse_targets_text(const std::string& txt,TargetSet& ts,bool addr,u64& lines,u64& rejected,u64& dups,u64& unsupported){
    lines=rejected=dups=unsupported=0; ts.addr=addr;
    size_t i=0;
    if(txt.size()>=3&&(unsigned char)txt[0]==0xEF&&(unsigned char)txt[1]==0xBB&&(unsigned char)txt[2]==0xBF) i=3;
    while(i<txt.size()){
        size_t j=txt.find('\n',i); if(j==std::string::npos) j=txt.size();
        size_t a=i,b=j;
        while(a<b&&(txt[a]==' '||txt[a]=='\t'||txt[a]=='\r')) a++;
        size_t t=a; while(t<b&&txt[t]!=' '&&txt[t]!='\t'&&txt[t]!='\r') t++;   // 1er mot de la ligne
        if(t>a){
            lines++;
            Ent e; std::string tok=txt.substr(a,t-a);
            if(addr){
                int rc=parse_address(tok,e);
                if(rc==1) ts.list.push_back(e); else if(rc==2) unsupported++; else rejected++;
            } else if(parse_pub(tok,e)) ts.list.push_back(e); else rejected++;
        }
        i=j+1;
    }
    u64 valid=ts.list.size();
    finalize_targets(ts);
    dups=valid-ts.list.size();
}
static bool read_file(const std::string& path,std::string& out){
    FILE* f=fopen(path.c_str(),"rb"); if(!f) return false;
    out.clear(); char buf[1<<16]; size_t n;
    while((n=fread(buf,1,sizeof buf,f))>0) out.append(buf,n);
    fclose(f); return true;
}
static u64 fnv64(const std::string& s){
    u64 h=1469598103934665603ULL;
    for(unsigned char c:s){ h^=c; h*=1099511628211ULL; }
    return h;
}
static u64 sum_hash(const void* p,size_t n,u64 h){
    const unsigned char* c=(const unsigned char*)p; size_t i=0;
    for(;i+8<=n;i+=8){ u64 w; memcpy(&w,c+i,8); h=(h^w)*0x9E3779B97F4A7C15ULL; h^=h>>29; }
    for(;i<n;i++){ h=(h^c[i])*0x100000001B3ULL; }
    return h;
}

// Fichier .bloom : en-tête (6 x u64) + filtre + liste exacte
static const char MAGIC[8]={'P','Z','B','L','O','O','M','2'};
static const u64 ENDIAN_TAG=0x0102030405060708ULL;
static u64 payload_sum(const TargetSet& ts){
    u64 h=0x1234567812345678ULL;
    h=sum_hash(ts.bloom.data(),ts.bloom.size()*8,h);
    h=sum_hash(ts.list.data(),ts.list.size()*sizeof(Ent),h);
    return h;
}
static bool save_bloom(const TargetSet& ts,const std::string& path){
    FILE* f=fopen(path.c_str(),"wb"); if(!f) return false;
    u64 hdr[6]={ENDIAN_TAG,(u64)ts.list.size(),ts.nblocks,ts.src_hash,payload_sum(ts),(u64)(ts.addr?1:0)};
    bool ok=fwrite(MAGIC,1,8,f)==8 && fwrite(hdr,8,6,f)==6
        && fwrite(ts.bloom.data(),8,ts.bloom.size(),f)==ts.bloom.size()
        && fwrite(ts.list.data(),sizeof(Ent),ts.list.size(),f)==ts.list.size();
    ok=(fclose(f)==0)&&ok;
    return ok;
}
static bool load_bloom(const std::string& path,TargetSet& ts,std::string& err){
    FILE* f=fopen(path.c_str(),"rb"); if(!f){ err="fichier introuvable"; return false; }
    char mg[8]; u64 hdr[6];
    if(fread(mg,1,8,f)!=8||memcmp(mg,MAGIC,8)!=0||fread(hdr,8,6,f)!=6){ fclose(f); err="format invalide"; return false; }
    if(hdr[0]!=ENDIAN_TAG){ fclose(f); err="format d'octets incompatible"; return false; }
    u64 n=hdr[1], nb=hdr[2];
    if(n==0||n>(1ULL<<32)||nb==0||(nb&(nb-1))||nb>(1ULL<<32)){ fclose(f); err="en-tête incohérent"; return false; }
    ts.nblocks=nb; ts.src_hash=hdr[3]; ts.addr=(hdr[5]==1);
    ts.bloom.assign(nb*8,0); ts.list.assign(n,Ent());
    bool ok=fread(ts.bloom.data(),8,ts.bloom.size(),f)==ts.bloom.size()
         && fread(ts.list.data(),sizeof(Ent),n,f)==n;
    char extra; if(ok && fread(&extra,1,1,f)==1) ok=false;          // fichier trop long
    fclose(f);
    if(!ok){ err="fichier tronqué ou corrompu"; return false; }
    if(payload_sum(ts)!=hdr[4]){ err="somme de contrôle invalide"; return false; }
    for(size_t i=0;i<n;i++){
        if(i&&!ent_less(ts.list[i-1],ts.list[i])){ err="liste non triée"; return false; }
        if(!bloom_test(ts.bloom.data(),nb-1,ts.list[i].x)){ err="filtre incohérent"; return false; }
    }
    return true;
}



// ================================================================ MOTEUR GPU (CUDA)
// Tout le code "périphérique" ci-dessous (préfixes g_ / gf_ / gj_) est écrit en C portable : il est compilé pour le GPU
// par nvcc ET pour l'hôte. Les auto-tests l'exécutent côté hôte (comparaison avec le code CPU de référence) puis sur le
// vrai GPU ; sans CUDA, le même noyau est exécuté en série sur le CPU (émulation) pour les tests.
// Arithmétique : éléments de Fp en 8 mots de 32 bits (little-endian), p = 2^256 - 2^32 - 977.
#ifndef GHALF
#define GHALF 128                      // un lot = 2*GHALF clés consécutives autour d'un point central (inversion par lots)
#endif
#define GTPB 128                       // threads par bloc CUDA
#define GBLK_PER_SM 4                  // blocs résidents par multiprocesseur (=> 512 threads/SM)
#define GMAXHIT (1u<<18)               // capacité du tampon de candidats (par GPU)
static_assert(GHALF%32==0 && GHALF>=32 && GHALF<=512,"GHALF doit etre un multiple de 32 (32..512)");

#define G_ROTR(x,n) (((x)>>(n))|((x)<<(32-(n))))
#define G_ROTL(x,n) (((x)<<(n))|((x)>>(32-(n))))
#define G_BSWAP(q) ((((q)>>24)&0xFFu)|(((q)>>8)&0xFF00u)|(((q)<<8)&0xFF0000u)|((q)<<24))
#define SHA_S0(x) (G_ROTR(x,2)^G_ROTR(x,13)^G_ROTR(x,22))
#define SHA_S1(x) (G_ROTR(x,6)^G_ROTR(x,11)^G_ROTR(x,25))
#define SHA_s0(x) (G_ROTR(x,7)^G_ROTR(x,18)^((x)>>3))
#define SHA_s1(x) (G_ROTR(x,17)^G_ROTR(x,19)^((x)>>10))
#define SHA_R(a,b,c,d,e,f,g,h,k,w) { u32 t1_=(h)+SHA_S1(e)+(((e)&(f))^(~(e)&(g)))+(k)+(w); \
    u32 t2_=SHA_S0(a)+(((a)&(b))^((a)&(c))^((b)&(c))); (d)+=t1_; (h)=t1_+t2_; }
#define RMD_F0(x,y,z) ((x)^(y)^(z))
#define RMD_F1(x,y,z) (((x)&(y))|(~(x)&(z)))
#define RMD_F2(x,y,z) (((x)|~(y))^(z))
#define RMD_F3(x,y,z) (((x)&(z))|((y)&~(z)))
#define RMD_F4(x,y,z) ((x)^((y)|~(z)))
#define RMD_STEP(f,a,b,c,d,e,x,k,s) { u32 t_=G_ROTL((a)+f(b,c,d)+(x)+(k),s)+(e); (a)=(e); (e)=(d); (d)=G_ROTL(c,10); (c)=(b); (b)=t_; }
// ---- SHA-256 (un bloc) : 64 tours déroulés, constantes littérales (généré)
HD static inline void g_sha256_blk(u32* w,u32* o){
    u32 a=0x6a09e667u,b=0xbb67ae85u,c=0x3c6ef372u,d=0xa54ff53au,e=0x510e527fu,f=0x9b05688cu,g=0x1f83d9abu,h=0x5be0cd19u;
    SHA_R(a,b,c,d,e,f,g,h,0x428a2f98u,w[0]);
    SHA_R(h,a,b,c,d,e,f,g,0x71374491u,w[1]);
    SHA_R(g,h,a,b,c,d,e,f,0xb5c0fbcfu,w[2]);
    SHA_R(f,g,h,a,b,c,d,e,0xe9b5dba5u,w[3]);
    SHA_R(e,f,g,h,a,b,c,d,0x3956c25bu,w[4]);
    SHA_R(d,e,f,g,h,a,b,c,0x59f111f1u,w[5]);
    SHA_R(c,d,e,f,g,h,a,b,0x923f82a4u,w[6]);
    SHA_R(b,c,d,e,f,g,h,a,0xab1c5ed5u,w[7]);
    SHA_R(a,b,c,d,e,f,g,h,0xd807aa98u,w[8]);
    SHA_R(h,a,b,c,d,e,f,g,0x12835b01u,w[9]);
    SHA_R(g,h,a,b,c,d,e,f,0x243185beu,w[10]);
    SHA_R(f,g,h,a,b,c,d,e,0x550c7dc3u,w[11]);
    SHA_R(e,f,g,h,a,b,c,d,0x72be5d74u,w[12]);
    SHA_R(d,e,f,g,h,a,b,c,0x80deb1feu,w[13]);
    SHA_R(c,d,e,f,g,h,a,b,0x9bdc06a7u,w[14]);
    SHA_R(b,c,d,e,f,g,h,a,0xc19bf174u,w[15]);
    w[0]+=SHA_s0(w[1])+w[9]+SHA_s1(w[14]);
    SHA_R(a,b,c,d,e,f,g,h,0xe49b69c1u,w[0]);
    w[1]+=SHA_s0(w[2])+w[10]+SHA_s1(w[15]);
    SHA_R(h,a,b,c,d,e,f,g,0xefbe4786u,w[1]);
    w[2]+=SHA_s0(w[3])+w[11]+SHA_s1(w[0]);
    SHA_R(g,h,a,b,c,d,e,f,0x0fc19dc6u,w[2]);
    w[3]+=SHA_s0(w[4])+w[12]+SHA_s1(w[1]);
    SHA_R(f,g,h,a,b,c,d,e,0x240ca1ccu,w[3]);
    w[4]+=SHA_s0(w[5])+w[13]+SHA_s1(w[2]);
    SHA_R(e,f,g,h,a,b,c,d,0x2de92c6fu,w[4]);
    w[5]+=SHA_s0(w[6])+w[14]+SHA_s1(w[3]);
    SHA_R(d,e,f,g,h,a,b,c,0x4a7484aau,w[5]);
    w[6]+=SHA_s0(w[7])+w[15]+SHA_s1(w[4]);
    SHA_R(c,d,e,f,g,h,a,b,0x5cb0a9dcu,w[6]);
    w[7]+=SHA_s0(w[8])+w[0]+SHA_s1(w[5]);
    SHA_R(b,c,d,e,f,g,h,a,0x76f988dau,w[7]);
    w[8]+=SHA_s0(w[9])+w[1]+SHA_s1(w[6]);
    SHA_R(a,b,c,d,e,f,g,h,0x983e5152u,w[8]);
    w[9]+=SHA_s0(w[10])+w[2]+SHA_s1(w[7]);
    SHA_R(h,a,b,c,d,e,f,g,0xa831c66du,w[9]);
    w[10]+=SHA_s0(w[11])+w[3]+SHA_s1(w[8]);
    SHA_R(g,h,a,b,c,d,e,f,0xb00327c8u,w[10]);
    w[11]+=SHA_s0(w[12])+w[4]+SHA_s1(w[9]);
    SHA_R(f,g,h,a,b,c,d,e,0xbf597fc7u,w[11]);
    w[12]+=SHA_s0(w[13])+w[5]+SHA_s1(w[10]);
    SHA_R(e,f,g,h,a,b,c,d,0xc6e00bf3u,w[12]);
    w[13]+=SHA_s0(w[14])+w[6]+SHA_s1(w[11]);
    SHA_R(d,e,f,g,h,a,b,c,0xd5a79147u,w[13]);
    w[14]+=SHA_s0(w[15])+w[7]+SHA_s1(w[12]);
    SHA_R(c,d,e,f,g,h,a,b,0x06ca6351u,w[14]);
    w[15]+=SHA_s0(w[0])+w[8]+SHA_s1(w[13]);
    SHA_R(b,c,d,e,f,g,h,a,0x14292967u,w[15]);
    w[0]+=SHA_s0(w[1])+w[9]+SHA_s1(w[14]);
    SHA_R(a,b,c,d,e,f,g,h,0x27b70a85u,w[0]);
    w[1]+=SHA_s0(w[2])+w[10]+SHA_s1(w[15]);
    SHA_R(h,a,b,c,d,e,f,g,0x2e1b2138u,w[1]);
    w[2]+=SHA_s0(w[3])+w[11]+SHA_s1(w[0]);
    SHA_R(g,h,a,b,c,d,e,f,0x4d2c6dfcu,w[2]);
    w[3]+=SHA_s0(w[4])+w[12]+SHA_s1(w[1]);
    SHA_R(f,g,h,a,b,c,d,e,0x53380d13u,w[3]);
    w[4]+=SHA_s0(w[5])+w[13]+SHA_s1(w[2]);
    SHA_R(e,f,g,h,a,b,c,d,0x650a7354u,w[4]);
    w[5]+=SHA_s0(w[6])+w[14]+SHA_s1(w[3]);
    SHA_R(d,e,f,g,h,a,b,c,0x766a0abbu,w[5]);
    w[6]+=SHA_s0(w[7])+w[15]+SHA_s1(w[4]);
    SHA_R(c,d,e,f,g,h,a,b,0x81c2c92eu,w[6]);
    w[7]+=SHA_s0(w[8])+w[0]+SHA_s1(w[5]);
    SHA_R(b,c,d,e,f,g,h,a,0x92722c85u,w[7]);
    w[8]+=SHA_s0(w[9])+w[1]+SHA_s1(w[6]);
    SHA_R(a,b,c,d,e,f,g,h,0xa2bfe8a1u,w[8]);
    w[9]+=SHA_s0(w[10])+w[2]+SHA_s1(w[7]);
    SHA_R(h,a,b,c,d,e,f,g,0xa81a664bu,w[9]);
    w[10]+=SHA_s0(w[11])+w[3]+SHA_s1(w[8]);
    SHA_R(g,h,a,b,c,d,e,f,0xc24b8b70u,w[10]);
    w[11]+=SHA_s0(w[12])+w[4]+SHA_s1(w[9]);
    SHA_R(f,g,h,a,b,c,d,e,0xc76c51a3u,w[11]);
    w[12]+=SHA_s0(w[13])+w[5]+SHA_s1(w[10]);
    SHA_R(e,f,g,h,a,b,c,d,0xd192e819u,w[12]);
    w[13]+=SHA_s0(w[14])+w[6]+SHA_s1(w[11]);
    SHA_R(d,e,f,g,h,a,b,c,0xd6990624u,w[13]);
    w[14]+=SHA_s0(w[15])+w[7]+SHA_s1(w[12]);
    SHA_R(c,d,e,f,g,h,a,b,0xf40e3585u,w[14]);
    w[15]+=SHA_s0(w[0])+w[8]+SHA_s1(w[13]);
    SHA_R(b,c,d,e,f,g,h,a,0x106aa070u,w[15]);
    w[0]+=SHA_s0(w[1])+w[9]+SHA_s1(w[14]);
    SHA_R(a,b,c,d,e,f,g,h,0x19a4c116u,w[0]);
    w[1]+=SHA_s0(w[2])+w[10]+SHA_s1(w[15]);
    SHA_R(h,a,b,c,d,e,f,g,0x1e376c08u,w[1]);
    w[2]+=SHA_s0(w[3])+w[11]+SHA_s1(w[0]);
    SHA_R(g,h,a,b,c,d,e,f,0x2748774cu,w[2]);
    w[3]+=SHA_s0(w[4])+w[12]+SHA_s1(w[1]);
    SHA_R(f,g,h,a,b,c,d,e,0x34b0bcb5u,w[3]);
    w[4]+=SHA_s0(w[5])+w[13]+SHA_s1(w[2]);
    SHA_R(e,f,g,h,a,b,c,d,0x391c0cb3u,w[4]);
    w[5]+=SHA_s0(w[6])+w[14]+SHA_s1(w[3]);
    SHA_R(d,e,f,g,h,a,b,c,0x4ed8aa4au,w[5]);
    w[6]+=SHA_s0(w[7])+w[15]+SHA_s1(w[4]);
    SHA_R(c,d,e,f,g,h,a,b,0x5b9cca4fu,w[6]);
    w[7]+=SHA_s0(w[8])+w[0]+SHA_s1(w[5]);
    SHA_R(b,c,d,e,f,g,h,a,0x682e6ff3u,w[7]);
    w[8]+=SHA_s0(w[9])+w[1]+SHA_s1(w[6]);
    SHA_R(a,b,c,d,e,f,g,h,0x748f82eeu,w[8]);
    w[9]+=SHA_s0(w[10])+w[2]+SHA_s1(w[7]);
    SHA_R(h,a,b,c,d,e,f,g,0x78a5636fu,w[9]);
    w[10]+=SHA_s0(w[11])+w[3]+SHA_s1(w[8]);
    SHA_R(g,h,a,b,c,d,e,f,0x84c87814u,w[10]);
    w[11]+=SHA_s0(w[12])+w[4]+SHA_s1(w[9]);
    SHA_R(f,g,h,a,b,c,d,e,0x8cc70208u,w[11]);
    w[12]+=SHA_s0(w[13])+w[5]+SHA_s1(w[10]);
    SHA_R(e,f,g,h,a,b,c,d,0x90befffau,w[12]);
    w[13]+=SHA_s0(w[14])+w[6]+SHA_s1(w[11]);
    SHA_R(d,e,f,g,h,a,b,c,0xa4506cebu,w[13]);
    w[14]+=SHA_s0(w[15])+w[7]+SHA_s1(w[12]);
    SHA_R(c,d,e,f,g,h,a,b,0xbef9a3f7u,w[14]);
    w[15]+=SHA_s0(w[0])+w[8]+SHA_s1(w[13]);
    SHA_R(b,c,d,e,f,g,h,a,0xc67178f2u,w[15]);
    o[0]=a+0x6a09e667u; o[1]=b+0xbb67ae85u; o[2]=c+0x3c6ef372u; o[3]=d+0xa54ff53au;
    o[4]=e+0x510e527fu; o[5]=f+0x9b05688cu; o[6]=g+0x1f83d9abu; o[7]=h+0x5be0cd19u;
}
// ---- RIPEMD-160 (un bloc, X = 16 mots little-endian) : 2 x 80 pas déroulés (généré)
HD static inline void g_rmd160_blk(const u32* X,u32* o){
    u32 a=0x67452301u,b=0xEFCDAB89u,c=0x98BADCFEu,d=0x10325476u,e=0xC3D2E1F0u;
    u32 a2=a,b2=b,c2=c,d2=d,e2=e;
    RMD_STEP(RMD_F0,a,b,c,d,e,X[0],0x00000000u,11);
    RMD_STEP(RMD_F4,a2,b2,c2,d2,e2,X[5],0x50a28be6u,8);
    RMD_STEP(RMD_F0,a,b,c,d,e,X[1],0x00000000u,14);
    RMD_STEP(RMD_F4,a2,b2,c2,d2,e2,X[14],0x50a28be6u,9);
    RMD_STEP(RMD_F0,a,b,c,d,e,X[2],0x00000000u,15);
    RMD_STEP(RMD_F4,a2,b2,c2,d2,e2,X[7],0x50a28be6u,9);
    RMD_STEP(RMD_F0,a,b,c,d,e,X[3],0x00000000u,12);
    RMD_STEP(RMD_F4,a2,b2,c2,d2,e2,X[0],0x50a28be6u,11);
    RMD_STEP(RMD_F0,a,b,c,d,e,X[4],0x00000000u,5);
    RMD_STEP(RMD_F4,a2,b2,c2,d2,e2,X[9],0x50a28be6u,13);
    RMD_STEP(RMD_F0,a,b,c,d,e,X[5],0x00000000u,8);
    RMD_STEP(RMD_F4,a2,b2,c2,d2,e2,X[2],0x50a28be6u,15);
    RMD_STEP(RMD_F0,a,b,c,d,e,X[6],0x00000000u,7);
    RMD_STEP(RMD_F4,a2,b2,c2,d2,e2,X[11],0x50a28be6u,15);
    RMD_STEP(RMD_F0,a,b,c,d,e,X[7],0x00000000u,9);
    RMD_STEP(RMD_F4,a2,b2,c2,d2,e2,X[4],0x50a28be6u,5);
    RMD_STEP(RMD_F0,a,b,c,d,e,X[8],0x00000000u,11);
    RMD_STEP(RMD_F4,a2,b2,c2,d2,e2,X[13],0x50a28be6u,7);
    RMD_STEP(RMD_F0,a,b,c,d,e,X[9],0x00000000u,13);
    RMD_STEP(RMD_F4,a2,b2,c2,d2,e2,X[6],0x50a28be6u,7);
    RMD_STEP(RMD_F0,a,b,c,d,e,X[10],0x00000000u,14);
    RMD_STEP(RMD_F4,a2,b2,c2,d2,e2,X[15],0x50a28be6u,8);
    RMD_STEP(RMD_F0,a,b,c,d,e,X[11],0x00000000u,15);
    RMD_STEP(RMD_F4,a2,b2,c2,d2,e2,X[8],0x50a28be6u,11);
    RMD_STEP(RMD_F0,a,b,c,d,e,X[12],0x00000000u,6);
    RMD_STEP(RMD_F4,a2,b2,c2,d2,e2,X[1],0x50a28be6u,14);
    RMD_STEP(RMD_F0,a,b,c,d,e,X[13],0x00000000u,7);
    RMD_STEP(RMD_F4,a2,b2,c2,d2,e2,X[10],0x50a28be6u,14);
    RMD_STEP(RMD_F0,a,b,c,d,e,X[14],0x00000000u,9);
    RMD_STEP(RMD_F4,a2,b2,c2,d2,e2,X[3],0x50a28be6u,12);
    RMD_STEP(RMD_F0,a,b,c,d,e,X[15],0x00000000u,8);
    RMD_STEP(RMD_F4,a2,b2,c2,d2,e2,X[12],0x50a28be6u,6);
    RMD_STEP(RMD_F1,a,b,c,d,e,X[7],0x5a827999u,7);
    RMD_STEP(RMD_F3,a2,b2,c2,d2,e2,X[6],0x5c4dd124u,9);
    RMD_STEP(RMD_F1,a,b,c,d,e,X[4],0x5a827999u,6);
    RMD_STEP(RMD_F3,a2,b2,c2,d2,e2,X[11],0x5c4dd124u,13);
    RMD_STEP(RMD_F1,a,b,c,d,e,X[13],0x5a827999u,8);
    RMD_STEP(RMD_F3,a2,b2,c2,d2,e2,X[3],0x5c4dd124u,15);
    RMD_STEP(RMD_F1,a,b,c,d,e,X[1],0x5a827999u,13);
    RMD_STEP(RMD_F3,a2,b2,c2,d2,e2,X[7],0x5c4dd124u,7);
    RMD_STEP(RMD_F1,a,b,c,d,e,X[10],0x5a827999u,11);
    RMD_STEP(RMD_F3,a2,b2,c2,d2,e2,X[0],0x5c4dd124u,12);
    RMD_STEP(RMD_F1,a,b,c,d,e,X[6],0x5a827999u,9);
    RMD_STEP(RMD_F3,a2,b2,c2,d2,e2,X[13],0x5c4dd124u,8);
    RMD_STEP(RMD_F1,a,b,c,d,e,X[15],0x5a827999u,7);
    RMD_STEP(RMD_F3,a2,b2,c2,d2,e2,X[5],0x5c4dd124u,9);
    RMD_STEP(RMD_F1,a,b,c,d,e,X[3],0x5a827999u,15);
    RMD_STEP(RMD_F3,a2,b2,c2,d2,e2,X[10],0x5c4dd124u,11);
    RMD_STEP(RMD_F1,a,b,c,d,e,X[12],0x5a827999u,7);
    RMD_STEP(RMD_F3,a2,b2,c2,d2,e2,X[14],0x5c4dd124u,7);
    RMD_STEP(RMD_F1,a,b,c,d,e,X[0],0x5a827999u,12);
    RMD_STEP(RMD_F3,a2,b2,c2,d2,e2,X[15],0x5c4dd124u,7);
    RMD_STEP(RMD_F1,a,b,c,d,e,X[9],0x5a827999u,15);
    RMD_STEP(RMD_F3,a2,b2,c2,d2,e2,X[8],0x5c4dd124u,12);
    RMD_STEP(RMD_F1,a,b,c,d,e,X[5],0x5a827999u,9);
    RMD_STEP(RMD_F3,a2,b2,c2,d2,e2,X[12],0x5c4dd124u,7);
    RMD_STEP(RMD_F1,a,b,c,d,e,X[2],0x5a827999u,11);
    RMD_STEP(RMD_F3,a2,b2,c2,d2,e2,X[4],0x5c4dd124u,6);
    RMD_STEP(RMD_F1,a,b,c,d,e,X[14],0x5a827999u,7);
    RMD_STEP(RMD_F3,a2,b2,c2,d2,e2,X[9],0x5c4dd124u,15);
    RMD_STEP(RMD_F1,a,b,c,d,e,X[11],0x5a827999u,13);
    RMD_STEP(RMD_F3,a2,b2,c2,d2,e2,X[1],0x5c4dd124u,13);
    RMD_STEP(RMD_F1,a,b,c,d,e,X[8],0x5a827999u,12);
    RMD_STEP(RMD_F3,a2,b2,c2,d2,e2,X[2],0x5c4dd124u,11);
    RMD_STEP(RMD_F2,a,b,c,d,e,X[3],0x6ed9eba1u,11);
    RMD_STEP(RMD_F2,a2,b2,c2,d2,e2,X[15],0x6d703ef3u,9);
    RMD_STEP(RMD_F2,a,b,c,d,e,X[10],0x6ed9eba1u,13);
    RMD_STEP(RMD_F2,a2,b2,c2,d2,e2,X[5],0x6d703ef3u,7);
    RMD_STEP(RMD_F2,a,b,c,d,e,X[14],0x6ed9eba1u,6);
    RMD_STEP(RMD_F2,a2,b2,c2,d2,e2,X[1],0x6d703ef3u,15);
    RMD_STEP(RMD_F2,a,b,c,d,e,X[4],0x6ed9eba1u,7);
    RMD_STEP(RMD_F2,a2,b2,c2,d2,e2,X[3],0x6d703ef3u,11);
    RMD_STEP(RMD_F2,a,b,c,d,e,X[9],0x6ed9eba1u,14);
    RMD_STEP(RMD_F2,a2,b2,c2,d2,e2,X[7],0x6d703ef3u,8);
    RMD_STEP(RMD_F2,a,b,c,d,e,X[15],0x6ed9eba1u,9);
    RMD_STEP(RMD_F2,a2,b2,c2,d2,e2,X[14],0x6d703ef3u,6);
    RMD_STEP(RMD_F2,a,b,c,d,e,X[8],0x6ed9eba1u,13);
    RMD_STEP(RMD_F2,a2,b2,c2,d2,e2,X[6],0x6d703ef3u,6);
    RMD_STEP(RMD_F2,a,b,c,d,e,X[1],0x6ed9eba1u,15);
    RMD_STEP(RMD_F2,a2,b2,c2,d2,e2,X[9],0x6d703ef3u,14);
    RMD_STEP(RMD_F2,a,b,c,d,e,X[2],0x6ed9eba1u,14);
    RMD_STEP(RMD_F2,a2,b2,c2,d2,e2,X[11],0x6d703ef3u,12);
    RMD_STEP(RMD_F2,a,b,c,d,e,X[7],0x6ed9eba1u,8);
    RMD_STEP(RMD_F2,a2,b2,c2,d2,e2,X[8],0x6d703ef3u,13);
    RMD_STEP(RMD_F2,a,b,c,d,e,X[0],0x6ed9eba1u,13);
    RMD_STEP(RMD_F2,a2,b2,c2,d2,e2,X[12],0x6d703ef3u,5);
    RMD_STEP(RMD_F2,a,b,c,d,e,X[6],0x6ed9eba1u,6);
    RMD_STEP(RMD_F2,a2,b2,c2,d2,e2,X[2],0x6d703ef3u,14);
    RMD_STEP(RMD_F2,a,b,c,d,e,X[13],0x6ed9eba1u,5);
    RMD_STEP(RMD_F2,a2,b2,c2,d2,e2,X[10],0x6d703ef3u,13);
    RMD_STEP(RMD_F2,a,b,c,d,e,X[11],0x6ed9eba1u,12);
    RMD_STEP(RMD_F2,a2,b2,c2,d2,e2,X[0],0x6d703ef3u,13);
    RMD_STEP(RMD_F2,a,b,c,d,e,X[5],0x6ed9eba1u,7);
    RMD_STEP(RMD_F2,a2,b2,c2,d2,e2,X[4],0x6d703ef3u,7);
    RMD_STEP(RMD_F2,a,b,c,d,e,X[12],0x6ed9eba1u,5);
    RMD_STEP(RMD_F2,a2,b2,c2,d2,e2,X[13],0x6d703ef3u,5);
    RMD_STEP(RMD_F3,a,b,c,d,e,X[1],0x8f1bbcdcu,11);
    RMD_STEP(RMD_F1,a2,b2,c2,d2,e2,X[8],0x7a6d76e9u,15);
    RMD_STEP(RMD_F3,a,b,c,d,e,X[9],0x8f1bbcdcu,12);
    RMD_STEP(RMD_F1,a2,b2,c2,d2,e2,X[6],0x7a6d76e9u,5);
    RMD_STEP(RMD_F3,a,b,c,d,e,X[11],0x8f1bbcdcu,14);
    RMD_STEP(RMD_F1,a2,b2,c2,d2,e2,X[4],0x7a6d76e9u,8);
    RMD_STEP(RMD_F3,a,b,c,d,e,X[10],0x8f1bbcdcu,15);
    RMD_STEP(RMD_F1,a2,b2,c2,d2,e2,X[1],0x7a6d76e9u,11);
    RMD_STEP(RMD_F3,a,b,c,d,e,X[0],0x8f1bbcdcu,14);
    RMD_STEP(RMD_F1,a2,b2,c2,d2,e2,X[3],0x7a6d76e9u,14);
    RMD_STEP(RMD_F3,a,b,c,d,e,X[8],0x8f1bbcdcu,15);
    RMD_STEP(RMD_F1,a2,b2,c2,d2,e2,X[11],0x7a6d76e9u,14);
    RMD_STEP(RMD_F3,a,b,c,d,e,X[12],0x8f1bbcdcu,9);
    RMD_STEP(RMD_F1,a2,b2,c2,d2,e2,X[15],0x7a6d76e9u,6);
    RMD_STEP(RMD_F3,a,b,c,d,e,X[4],0x8f1bbcdcu,8);
    RMD_STEP(RMD_F1,a2,b2,c2,d2,e2,X[0],0x7a6d76e9u,14);
    RMD_STEP(RMD_F3,a,b,c,d,e,X[13],0x8f1bbcdcu,9);
    RMD_STEP(RMD_F1,a2,b2,c2,d2,e2,X[5],0x7a6d76e9u,6);
    RMD_STEP(RMD_F3,a,b,c,d,e,X[3],0x8f1bbcdcu,14);
    RMD_STEP(RMD_F1,a2,b2,c2,d2,e2,X[12],0x7a6d76e9u,9);
    RMD_STEP(RMD_F3,a,b,c,d,e,X[7],0x8f1bbcdcu,5);
    RMD_STEP(RMD_F1,a2,b2,c2,d2,e2,X[2],0x7a6d76e9u,12);
    RMD_STEP(RMD_F3,a,b,c,d,e,X[15],0x8f1bbcdcu,6);
    RMD_STEP(RMD_F1,a2,b2,c2,d2,e2,X[13],0x7a6d76e9u,9);
    RMD_STEP(RMD_F3,a,b,c,d,e,X[14],0x8f1bbcdcu,8);
    RMD_STEP(RMD_F1,a2,b2,c2,d2,e2,X[9],0x7a6d76e9u,12);
    RMD_STEP(RMD_F3,a,b,c,d,e,X[5],0x8f1bbcdcu,6);
    RMD_STEP(RMD_F1,a2,b2,c2,d2,e2,X[7],0x7a6d76e9u,5);
    RMD_STEP(RMD_F3,a,b,c,d,e,X[6],0x8f1bbcdcu,5);
    RMD_STEP(RMD_F1,a2,b2,c2,d2,e2,X[10],0x7a6d76e9u,15);
    RMD_STEP(RMD_F3,a,b,c,d,e,X[2],0x8f1bbcdcu,12);
    RMD_STEP(RMD_F1,a2,b2,c2,d2,e2,X[14],0x7a6d76e9u,8);
    RMD_STEP(RMD_F4,a,b,c,d,e,X[4],0xa953fd4eu,9);
    RMD_STEP(RMD_F0,a2,b2,c2,d2,e2,X[12],0x00000000u,8);
    RMD_STEP(RMD_F4,a,b,c,d,e,X[0],0xa953fd4eu,15);
    RMD_STEP(RMD_F0,a2,b2,c2,d2,e2,X[15],0x00000000u,5);
    RMD_STEP(RMD_F4,a,b,c,d,e,X[5],0xa953fd4eu,5);
    RMD_STEP(RMD_F0,a2,b2,c2,d2,e2,X[10],0x00000000u,12);
    RMD_STEP(RMD_F4,a,b,c,d,e,X[9],0xa953fd4eu,11);
    RMD_STEP(RMD_F0,a2,b2,c2,d2,e2,X[4],0x00000000u,9);
    RMD_STEP(RMD_F4,a,b,c,d,e,X[7],0xa953fd4eu,6);
    RMD_STEP(RMD_F0,a2,b2,c2,d2,e2,X[1],0x00000000u,12);
    RMD_STEP(RMD_F4,a,b,c,d,e,X[12],0xa953fd4eu,8);
    RMD_STEP(RMD_F0,a2,b2,c2,d2,e2,X[5],0x00000000u,5);
    RMD_STEP(RMD_F4,a,b,c,d,e,X[2],0xa953fd4eu,13);
    RMD_STEP(RMD_F0,a2,b2,c2,d2,e2,X[8],0x00000000u,14);
    RMD_STEP(RMD_F4,a,b,c,d,e,X[10],0xa953fd4eu,12);
    RMD_STEP(RMD_F0,a2,b2,c2,d2,e2,X[7],0x00000000u,6);
    RMD_STEP(RMD_F4,a,b,c,d,e,X[14],0xa953fd4eu,5);
    RMD_STEP(RMD_F0,a2,b2,c2,d2,e2,X[6],0x00000000u,8);
    RMD_STEP(RMD_F4,a,b,c,d,e,X[1],0xa953fd4eu,12);
    RMD_STEP(RMD_F0,a2,b2,c2,d2,e2,X[2],0x00000000u,13);
    RMD_STEP(RMD_F4,a,b,c,d,e,X[3],0xa953fd4eu,13);
    RMD_STEP(RMD_F0,a2,b2,c2,d2,e2,X[13],0x00000000u,6);
    RMD_STEP(RMD_F4,a,b,c,d,e,X[8],0xa953fd4eu,14);
    RMD_STEP(RMD_F0,a2,b2,c2,d2,e2,X[14],0x00000000u,5);
    RMD_STEP(RMD_F4,a,b,c,d,e,X[11],0xa953fd4eu,11);
    RMD_STEP(RMD_F0,a2,b2,c2,d2,e2,X[0],0x00000000u,15);
    RMD_STEP(RMD_F4,a,b,c,d,e,X[6],0xa953fd4eu,8);
    RMD_STEP(RMD_F0,a2,b2,c2,d2,e2,X[3],0x00000000u,13);
    RMD_STEP(RMD_F4,a,b,c,d,e,X[15],0xa953fd4eu,5);
    RMD_STEP(RMD_F0,a2,b2,c2,d2,e2,X[9],0x00000000u,11);
    RMD_STEP(RMD_F4,a,b,c,d,e,X[13],0xa953fd4eu,6);
    RMD_STEP(RMD_F0,a2,b2,c2,d2,e2,X[11],0x00000000u,11);
    o[0]=0xEFCDAB89u+c+d2; o[1]=0x98BADCFEu+d+e2; o[2]=0x10325476u+e+a2; o[3]=0xC3D2E1F0u+a+b2; o[4]=0x67452301u+b+c2;
}
// hash160 d'une clé publique compressée : x = X en 8 mots little-endian, par = parité de Y -> h[0..4] (même ordre que pack_rmd)
HD static inline void g_hash160(const u32* xv,u32 par,u32* h){
    u32 w[16],H[8],rw[16];
    w[0]=((0x02u+par)<<24)|(xv[7]>>8);
    for(int i=1;i<8;i++) w[i]=(xv[8-i]<<24)|(xv[7-i]>>8);
    w[8]=(xv[0]<<24)|0x00800000u;
    for(int i=9;i<15;i++) w[i]=0;
    w[15]=264;
    g_sha256_blk(w,H);
    for(int i=0;i<8;i++) rw[i]=G_BSWAP(H[i]);
    rw[8]=0x80; for(int i=9;i<14;i++) rw[i]=0; rw[14]=256; rw[15]=0;
    g_rmd160_blk(rw,h);
}

// ---------------------------------------------------------------- corps Fp en 32 bits
struct FE { u32 v[8]; };
HD static inline void gf_zero(FE& r){ for(int i=0;i<8;i++) r.v[i]=0; }
HD static inline void gf_one(FE& r){ r.v[0]=1; for(int i=1;i<8;i++) r.v[i]=0; }
HD static inline bool gf_is_zero(const FE& a){ return (a.v[0]|a.v[1]|a.v[2]|a.v[3]|a.v[4]|a.v[5]|a.v[6]|a.v[7])==0; }
HD static inline FE gf_from_u256(const U256& a){ FE r; for(int i=0;i<4;i++){ r.v[2*i]=(u32)a.d[i]; r.v[2*i+1]=(u32)(a.d[i]>>32); } return r; }
HD static inline U256 gf_to_u256(const FE& a){ U256 r; for(int i=0;i<4;i++) r.d[i]=(u64)a.v[2*i]|((u64)a.v[2*i+1]<<32); return r; }

HD static inline void gf_add(FE& r,const FE& a,const FE& b){
    u32 t[8],u[8]; u64 c=0;
    #pragma unroll
    for(int i=0;i<8;i++){ c+=(u64)a.v[i]+b.v[i]; t[i]=(u32)c; c>>=32; }
    const u32 c1=(u32)c;
    c=0;
    #pragma unroll
    for(int i=0;i<8;i++){ c+=(u64)t[i]+(i==0?0x3D1u:(i==1?1u:0u)); u[i]=(u32)c; c>>=32; }
    const u32 sel=0u-(c1|(u32)c);                    // 0xFFFFFFFF : le résultat est t + (2^32+977) mod 2^256
    #pragma unroll
    for(int i=0;i<8;i++) r.v[i]=(u[i]&sel)|(t[i]&~sel);
}
HD static inline void gf_sub(FE& r,const FE& a,const FE& b){
    u32 t[8]; u64 br=0;
    #pragma unroll
    for(int i=0;i<8;i++){ u64 d=(u64)a.v[i]-b.v[i]-br; t[i]=(u32)d; br=(d>>63)&1; }
    const u32 m=0u-(u32)br;                          // emprunt : on retranche 2^32+977 (= ajouter p)
    u64 bo=0;
    #pragma unroll
    for(int i=0;i<8;i++){ u64 k=(i==0)?(u64)(0x3D1u&m):((i==1)?(u64)(1u&m):0ull); u64 d=(u64)t[i]-k-bo; r.v[i]=(u32)d; bo=(d>>63)&1; }
}
// Réduction d'un produit de 512 bits (16 mots) modulo p : 2^256 = 2^32 + 977 (mod p)
HD static inline void gf_reduce(FE& r,const u32* t){
    u32 s[8],u[8]; u64 c=0;
    #pragma unroll
    for(int i=0;i<8;i++){
        u64 cur=(u64)t[8+i]*977u+t[i]+c;
        if(i>0) cur+=t[7+i];
        s[i]=(u32)cur; c=cur>>32;
    }
    const u64 top=c+(u64)t[15];                      // < 2^34 : poids 2^256
    u64 acc=(u64)s[0]+top*977u; s[0]=(u32)acc; acc>>=32;
    acc+=(u64)s[1]+top; s[1]=(u32)acc; acc>>=32;
    #pragma unroll
    for(int i=2;i<8;i++){ acc+=s[i]; s[i]=(u32)acc; acc>>=32; }
    {   const u32 m=0u-(u32)acc;                     // retenue 2^256 : on la replie (valeur alors minuscule)
        acc=(u64)s[0]+(0x3D1u&m); s[0]=(u32)acc; acc>>=32;
        acc+=(u64)s[1]+(1u&m); s[1]=(u32)acc; acc>>=32;
        #pragma unroll
        for(int i=2;i<8;i++){ acc+=s[i]; s[i]=(u32)acc; acc>>=32; }
    }
    acc=0;                                           // forme canonique : si s >= p alors s-p = s + (2^32+977) - 2^256
    #pragma unroll
    for(int i=0;i<8;i++){ acc+=(u64)s[i]+(i==0?0x3D1u:(i==1?1u:0u)); u[i]=(u32)acc; acc>>=32; }
    const u32 sel=0u-(u32)acc;
    #pragma unroll
    for(int i=0;i<8;i++) r.v[i]=(u[i]&sel)|(s[i]&~sel);
}
HD static inline void gf_mul(FE& r,const FE& a,const FE& b){
    u32 t[16];
    #pragma unroll
    for(int i=0;i<16;i++) t[i]=0;
    #pragma unroll
    for(int i=0;i<8;i++){
        u64 carry=0;
        #pragma unroll
        for(int j=0;j<8;j++){
            u64 cur=(u64)a.v[i]*b.v[j]+t[i+j]+carry;
            t[i+j]=(u32)cur; carry=cur>>32;
        }
        t[i+8]=(u32)carry;
    }
    gf_reduce(r,t);
}
// Carré dédié : 36 produits de 32 bits au lieu de 64
HD static inline void gf_sqr(FE& r,const FE& a){
    u32 t[16];
    #pragma unroll
    for(int i=0;i<16;i++) t[i]=0;
    #pragma unroll
    for(int i=0;i<7;i++){
        u64 carry=0;
        #pragma unroll
        for(int j=i+1;j<8;j++){
            u64 cur=(u64)a.v[i]*a.v[j]+t[i+j]+carry;
            t[i+j]=(u32)cur; carry=cur>>32;
        }
        t[i+8]=(u32)carry;
    }
    u32 c=0;
    #pragma unroll
    for(int k=0;k<16;k++){ u32 v=t[k]; t[k]=(v<<1)|c; c=v>>31; }
    u64 carry=0;
    #pragma unroll
    for(int i=0;i<8;i++){
        u64 sq=(u64)a.v[i]*a.v[i];
        u64 cur=(u64)t[2*i]+(u32)sq+carry;
        t[2*i]=(u32)cur; carry=cur>>32;
        cur=(u64)t[2*i+1]+(u32)(sq>>32)+carry;
        t[2*i+1]=(u32)cur; carry=cur>>32;
    }
    gf_reduce(r,t);
}
HD static inline void gf_sqr_n(FE& r,const FE& a,int n){
    FE t=a;
    #pragma unroll 1
    for(int i=0;i<n;i++) gf_sqr(t,t);
    r=t;
}
// Inverse modulaire a^(p-2) : 255 carrés + 15 produits
HD static void gf_inv(FE& r,const FE& a){
    FE x2,x3,x6,x9,x11,x22,x44,x88,x176,x220,x223,t;
    gf_sqr(x2,a); gf_mul(x2,x2,a);
    gf_sqr(x3,x2); gf_mul(x3,x3,a);
    gf_sqr_n(x6,x3,3); gf_mul(x6,x6,x3);
    gf_sqr_n(x9,x6,3); gf_mul(x9,x9,x3);
    gf_sqr_n(x11,x9,2); gf_mul(x11,x11,x2);
    gf_sqr_n(x22,x11,11); gf_mul(x22,x22,x11);
    gf_sqr_n(x44,x22,22); gf_mul(x44,x44,x22);
    gf_sqr_n(x88,x44,44); gf_mul(x88,x88,x44);
    gf_sqr_n(x176,x88,88); gf_mul(x176,x176,x88);
    gf_sqr_n(x220,x176,44); gf_mul(x220,x220,x44);
    gf_sqr_n(x223,x220,3); gf_mul(x223,x223,x3);
    gf_sqr_n(t,x223,23); gf_mul(t,t,x22);
    gf_sqr_n(t,t,5); gf_mul(t,t,a);
    gf_sqr_n(t,t,3); gf_mul(t,t,x2);
    gf_sqr_n(t,t,2); gf_mul(t,t,a);
    r=t;
}

// ---------------------------------------------------------------- tables en mémoire constante (GPU) / mémoire hôte (émulation, tests)
struct GTables { u32 gx[GHALF+1][8]; u32 gy[GHALF+1][8]; u32 sst[2][8]; u32 gen[2][8]; };
static GTables h_tab;                                  // copie hôte (remplie par gpu_prepare_tables)
#ifdef __CUDACC__
__constant__ GTables c_tab;
#endif
#if defined(__CUDA_ARCH__)
  #define TAB(f) c_tab.f
#else
  #define TAB(f) h_tab.f
#endif
HD static inline void g_ld(FE& r,const u32* p){ for(int k=0;k<8;k++) r.v[k]=p[k]; }

// ---------------------------------------------------------------- courbe : coordonnées jacobiennes
struct GJac { FE X,Y,Z; bool inf; };
HD static inline void gj_set_inf(GJac& R){ gf_zero(R.X); gf_zero(R.Y); gf_zero(R.Z); R.inf=true; }
HD static void gj_dbl(GJac& R,const GJac& P){
    if(P.inf||gf_is_zero(P.Y)){ gj_set_inf(R); return; }
    FE A,B,C,D,E,F,t,t2,X3,Y3,Z3;
    gf_sqr(A,P.X); gf_sqr(B,P.Y); gf_sqr(C,B);
    gf_add(t,P.X,B); gf_sqr(t,t); gf_sub(t,t,A); gf_sub(t,t,C); gf_add(D,t,t);
    gf_add(E,A,A); gf_add(E,E,A);
    gf_sqr(F,E);
    gf_add(t,D,D); gf_sub(X3,F,t);
    gf_sub(t,D,X3); gf_mul(t,E,t);
    gf_add(t2,C,C); gf_add(t2,t2,t2); gf_add(t2,t2,t2);
    gf_sub(Y3,t,t2);
    gf_mul(t,P.Y,P.Z); gf_add(Z3,t,t);
    R.X=X3; R.Y=Y3; R.Z=Z3; R.inf=false;
}
HD static void gj_madd(GJac& R,const GJac& P,const FE& qx,const FE& qy){
    if(P.inf){ R.X=qx; R.Y=qy; gf_one(R.Z); R.inf=false; return; }
    FE Z1Z1,U2,S2,H,HH,I,J,r,V,t,u,X3,Y3,Z3;
    gf_sqr(Z1Z1,P.Z); gf_mul(U2,qx,Z1Z1);
    gf_mul(S2,qy,P.Z); gf_mul(S2,S2,Z1Z1);
    gf_sub(H,U2,P.X);
    gf_sub(t,S2,P.Y); gf_add(r,t,t);
    if(gf_is_zero(H)){
        if(gf_is_zero(r)){ gj_dbl(R,P); return; }
        gj_set_inf(R); return;
    }
    gf_sqr(HH,H); gf_add(I,HH,HH); gf_add(I,I,I);
    gf_mul(J,H,I); gf_mul(V,P.X,I);
    gf_sqr(X3,r); gf_sub(X3,X3,J); gf_sub(X3,X3,V); gf_sub(X3,X3,V);
    gf_sub(t,V,X3); gf_mul(t,r,t);
    gf_mul(u,P.Y,J); gf_add(u,u,u);
    gf_sub(Y3,t,u);
    gf_add(t,P.Z,H); gf_sqr(t,t); gf_sub(t,t,Z1Z1); gf_sub(Z3,t,HH);
    R.X=X3; R.Y=Y3; R.Z=Z3; R.inf=false;
}
// k*G en coordonnées affines (k >= 1)
HD static void g_point_of_key(const U256& k,FE& x,FE& y){
    FE gx,gy; g_ld(gx,TAB(gen)[0]); g_ld(gy,TAB(gen)[1]);
    GJac R; gj_set_inf(R);
    int top=255; while(top>=0 && !((k.d[top>>6]>>(top&63))&1)) top--;
    for(int i=top;i>=0;--i){
        gj_dbl(R,R);
        if((k.d[i>>6]>>(i&63))&1) gj_madd(R,R,gx,gy);
    }
    if(R.inf){ gf_zero(x); gf_zero(y); return; }
    FE zi,zi2,zi3; gf_inv(zi,R.Z); gf_sqr(zi2,zi); gf_mul(zi3,zi2,zi);
    gf_mul(x,R.X,zi2); gf_mul(y,R.Y,zi3);
}

// ---------------------------------------------------------------- clés 256 bits (4 x u64) : opérations sans __int128
HD static inline void k_add(U256& a,u64 v){ u64 c=v; for(int i=0;i<4;i++){ u64 s=a.d[i]+c; c=(s<c)?1ull:0ull; a.d[i]=s; } }
HD static inline void k_sub(U256& a,u64 v){ u64 b=v; for(int i=0;i<4;i++){ u64 s=a.d[i]-b; b=(a.d[i]<b)?1ull:0ull; a.d[i]=s; } }
HD static inline void k_sub256(U256& r,const U256& a,const U256& b){
    u64 br=0; for(int i=0;i<4;i++){ u64 s=a.d[i]-b.d[i]-br; br=(a.d[i]<b.d[i]||(a.d[i]==b.d[i]&&br))?1ull:0ull; r.d[i]=s; }
}

// ---------------------------------------------------------------- noyau de recherche
struct GHit { u64 key[4]; u64 val[4]; u64 par; };       // clé candidate ; val = X (clé publique) ou hash160 empaqueté ; par=2 : à vérifier d'office
struct GArgs {
    u32* st; u64* lo; u64* hi; u64* cen; u32* cenok;    // état persistant par thread
    u32* asg; u64* nlo; u64* nhi;                       // nouvelles affectations de blocs (écrites par l'hôte)
    const u64* bloom; u64 bmask;                        // filtre de Bloom des cibles
    GHit* hits; u32* nhit; u32 maxhit; u32 guard;       // candidats
    unsigned long long* kdone;                          // clés testées (cumul)
    int nthr; int budget;                               // threads ; lots (2*GHALF clés) par thread et par lancement
};
HD static inline u32 g_atomic_inc(u32* p){
#if defined(__CUDA_ARCH__)
    return atomicAdd(p,1u);
#else
    return (*p)++;                                      // émulation : exécution en série
#endif
}
HD static inline void g_atomic_add64(unsigned long long* p,unsigned long long v){
#if defined(__CUDA_ARCH__)
    atomicAdd(p,v);
#else
    *p+=v;
#endif
}
HD static inline bool g_bloom_test(const u64* bl,u64 mask,u64 x0,u64 x1){
    const u64* blk=bl+((x0&mask)<<3);
    for(int j=0;j<6;j++){ u64 p=(x1>>(9*j))&511; if(!((blk[p>>6]>>(p&63))&1)) return false; }
    return true;
}
HD static inline void g_key_of(U256& k,const U256& kc,int off){ k=kc; if(off>=0) k_add(k,(u64)off); else k_sub(k,(u64)(-off)); }
HD static inline void g_record(const GArgs& A,const U256& kc,int off,u64 v0,u64 v1,u64 v2,u64 v3,u64 par){
    u32 idx=g_atomic_inc(A.nhit);
    if(idx<A.maxhit){
        GHit& h=A.hits[idx]; U256 k; g_key_of(k,kc,off);
        for(int i=0;i<4;i++) h.key[i]=k.d[i];
        h.val[0]=v0; h.val[1]=v1; h.val[2]=v2; h.val[3]=v3; h.par=par;
    }
}
// Teste une clé (point x, parité de y) contre le filtre ; si elle passe, elle est notée pour vérification exacte par l'hôte
template<bool ADDR>
HD static inline void g_emit(const GArgs& A,const U256& kc,int off,const FE& x,u32 par){
    if(ADDR){
        u32 h[5]; g_hash160(x.v,par,h);
        const u64 d0=(u64)h[0]|((u64)h[1]<<32), d1=(u64)h[2]|((u64)h[3]<<32);
        if(g_bloom_test(A.bloom,A.bmask,d0,d1)) g_record(A,kc,off,d0,d1,(u64)h[4],0,0);
    } else {
        const u64 d0=(u64)x.v[0]|((u64)x.v[1]<<32), d1=(u64)x.v[2]|((u64)x.v[3]<<32);
        if(g_bloom_test(A.bloom,A.bmask,d0,d1))
            g_record(A,kc,off,d0,d1,(u64)x.v[4]|((u64)x.v[5]<<32),(u64)x.v[6]|((u64)x.v[7]<<32),0);
    }
}
// Un lot : clés kc-GHALF .. kc+GHALF-1 (point central C = kc*G). Même algorithme que le CPU : une inversion par lots.
template<bool ADDR>
HD static void g_group(const GArgs& A,const U256& kc,const FE& Cx,const FE& Cy,bool last,FE& Cnx,FE& Cny,bool& have_next){
    FE pre[GHALF+1];
    u32 zm[(GHALF>>5)+1];
    for(int w=0;w<(GHALF>>5)+1;w++) zm[w]=0;
    FE gxi,gyi,sstx,ssty,acc,inv,invd,d,t,l,x3,y3;
    g_emit<ADDR>(A,kc,0,Cx,Cy.v[0]&1u);                                    // clé centrale
    gf_one(acc);
    #pragma unroll 1
    for(int i=1;i<=GHALF;i++){
        g_ld(gxi,TAB(gx)[i]); gf_sub(d,gxi,Cx);
        if(gf_is_zero(d)){ zm[i>>5]|=1u<<(i&31); gf_one(d); }
        gf_mul(acc,acc,d); pre[i]=acc;
    }
    g_ld(sstx,TAB(sst)[0]); g_ld(ssty,TAB(sst)[1]);
    FE dS; gf_sub(dS,sstx,Cx);
    const bool zs=gf_is_zero(dS); if(zs) gf_one(dS);
    gf_mul(acc,acc,dS);
    gf_inv(inv,acc);
    have_next=false;
    gf_mul(invd,inv,pre[GHALF]); gf_mul(inv,inv,dS);
    if(!zs&&!last){                                                         // centre du lot suivant : C + 2*GHALF*G
        gf_sub(t,ssty,Cy); gf_mul(l,t,invd);
        gf_sqr(Cnx,l); gf_sub(Cnx,Cnx,Cx); gf_sub(Cnx,Cnx,sstx);
        gf_sub(t,Cx,Cnx); gf_mul(Cny,l,t); gf_sub(Cny,Cny,Cy);
        have_next=true;
    }
    #pragma unroll 1
    for(int i=GHALF;i>=1;--i){
        g_ld(gxi,TAB(gx)[i]);
        const bool zi=((zm[i>>5]>>(i&31))&1u)!=0;
        if(i>1){
            gf_mul(invd,inv,pre[i-1]);
            gf_sub(d,gxi,Cx); if(zi) gf_one(d);
            gf_mul(inv,inv,d);
        } else invd=inv;
        if(zi){                                                             // cas dégénéré (C = ±i*G) : vérification exacte par l'hôte
            if(i<GHALF) g_record(A,kc,i,0,0,0,0,2);
            g_record(A,kc,-i,0,0,0,0,2);
            continue;
        }
        g_ld(gyi,TAB(gy)[i]);
        u32 par=0;
        gf_add(t,gyi,Cy); gf_mul(l,t,invd);                                 // clé kc-i : C + (-iG)
        gf_sqr(x3,l); gf_sub(x3,x3,Cx); gf_sub(x3,x3,gxi);
        if(ADDR){ gf_sub(t,x3,Cx); gf_mul(y3,l,t); gf_sub(y3,y3,Cy); par=y3.v[0]&1u; }
        g_emit<ADDR>(A,kc,-i,x3,par);
        if(i<GHALF){                                                        // clé kc+i : C + iG
            gf_sub(t,gyi,Cy); gf_mul(l,t,invd);
            gf_sqr(x3,l); gf_sub(x3,x3,Cx); gf_sub(x3,x3,gxi);
            if(ADDR){ gf_sub(t,Cx,x3); gf_mul(y3,l,t); gf_sub(y3,y3,Cy); par=y3.v[0]&1u; }
            g_emit<ADDR>(A,kc,i,x3,par);
        }
    }
}
// Travail d'un thread pendant un lancement : au plus A.budget lots, puis l'état est sauvegardé (le bloc continue au lancement suivant)
template<bool ADDR>
HD static void g_thread(int t,const GArgs& A){
    if(A.asg[t]){                                                           // nouveau bloc [nlo, nhi]
        for(int k=0;k<4;k++){ A.lo[t*4+k]=A.nlo[t*4+k]; A.hi[t*4+k]=A.nhi[t*4+k]; }
        A.st[t]=1; A.cenok[t]=0; A.asg[t]=0;
    }
    if(!A.st[t]) return;
    U256 lo,hi; for(int k=0;k<4;k++){ lo.d[k]=A.lo[t*4+k]; hi.d[k]=A.hi[t*4+k]; }
    U256 kc=lo; k_add(kc,(u64)GHALF);
    FE Cx,Cy;
    if(A.cenok[t]){ for(int k=0;k<4;k++){ u64 a=A.cen[t*8+k], b=A.cen[t*8+4+k]; Cx.v[2*k]=(u32)a; Cx.v[2*k+1]=(u32)(a>>32); Cy.v[2*k]=(u32)b; Cy.v[2*k+1]=(u32)(b>>32); } }
    else g_point_of_key(kc,Cx,Cy);
    u64 nk=0; bool fin=false;
    for(int g=0;g<A.budget;g++){
        if(*(volatile u32*)A.nhit>=A.guard) break;                          // tampon presque plein : on reprend au lancement suivant
        U256 diff; k_sub256(diff,hi,lo);
        const bool last=((diff.d[1]|diff.d[2]|diff.d[3])==0)&&diff.d[0]<(u64)(2*GHALF);
        const u64 count=last? diff.d[0]+1 : (u64)(2*GHALF);
        FE Cnx,Cny; bool have_next=false;
        g_group<ADDR>(A,kc,Cx,Cy,last,Cnx,Cny,have_next);
        nk+=count;
        if(last){ fin=true; break; }
        k_add(lo,(u64)(2*GHALF)); k_add(kc,(u64)(2*GHALF));
        if(have_next){ Cx=Cnx; Cy=Cny; } else g_point_of_key(kc,Cx,Cy);
    }
    if(fin) A.st[t]=0;
    else{
        for(int k=0;k<4;k++){
            A.lo[t*4+k]=lo.d[k];
            A.cen[t*8+k]=(u64)Cx.v[2*k]|((u64)Cx.v[2*k+1]<<32);
            A.cen[t*8+4+k]=(u64)Cy.v[2*k]|((u64)Cy.v[2*k+1]<<32);
        }
        A.cenok[t]=1;
    }
    if(nk) g_atomic_add64(A.kdone,(unsigned long long)nk);
}
#ifdef __CUDACC__
template<bool ADDR>
__global__ void __launch_bounds__(GTPB,GBLK_PER_SM) g_kernel(GArgs A){
    const int t=(int)(blockIdx.x*blockDim.x+threadIdx.x);
    if(t<A.nthr) g_thread<ADDR>(t,A);
}
#endif

// ---------------------------------------------------------------- recherche : état partagé
static const int HALF=512;                 // chaque lot CPU teste 2*HALF clés consécutives autour d'un point central
static std::vector<U256> gx,gy;            // i*G pour i = 1..HALF
static Aff SST;                            // (2*HALF)*G : pas vers le lot suivant
static bool g_tables_built=false;

static void build_tables(){
    if(g_tables_built) return;
    gx.assign(HALF+1,ZERO); gy.assign(HALF+1,ZERO);
    Aff cur=GEN;
    for(int i=1;i<=HALF;i++){ gx[i]=cur.x; gy[i]=cur.y; if(i<HALF) cur=aff_add(cur,GEN); }
    SST=aff_add(cur,cur);                  // cur = HALF*G
    g_tables_built=true;
}

struct Found { U256 key; Ent ent; std::string pub; double t=0; bool saved=false; bool outside=false; std::string path; };
struct TState {
    std::atomic<u64> keys; std::atomic<u64> block; std::atomic<int> mode;
    SpinLock lk; bool iact; u64 iblk,ioff;                // bloc en cours et position (pour le point de reprise)
    char pad[40];
    TState():keys(0),block(0),mode(-1),iact(false),iblk(0),ioff(0){}
};
// Suivi des blocs en cours d'un GPU (un par thread GPU)
struct GpuTrack {
    SpinLock lk; std::vector<u64> blk,off; std::vector<char> act;
    std::atomic<int> nthr, busy;
    GpuTrack():nthr(0),busy(0){}
};
typedef std::chrono::steady_clock clk;

// Modes de parcours des blocs (un seul par lancement, tous les cœurs travaillent sur la même suite)
enum Mode { MD_SOBOL=0, MD_RANDOM=1, MD_SEQ=2, MD_MC=3, MD_PI=4, MD_E=5, MD_LN=6, MD_COUNT=7 };
static const char* MODE_NAME[MD_COUNT]={"Sobol","Aléatoire","Séquentiel","Monte Carlo","Pi","e","ln 2"};
// parties fractionnaires de pi, e et ln(2) sur 64 bits (suites de Weyl : bloc = frac(i*c) * nombre_de_blocs)
static const u64 WEYL_C[3]={0x243F6A8885A308D3ULL,0xB7E151628AED2A6BULL,0xB17217F7D1CF79ABULL};

static long double u256_to_ld(const U256& a){
    long double r=0, m=1; const long double B=18446744073709551616.0L;
    for(int i=0;i<4;i++){ r+=(long double)a.d[i]*m; m*=B; }
    return r;
}
// Ordre n de la courbe secp256k1 : toute clé privée valide est dans [1, n-1]
static const U256 CURVE_N={{0xBFD25E8CD0364141ULL,0xBAAEDCE6AF48A03BULL,0xFFFFFFFFFFFFFFFEULL,0xFFFFFFFFFFFFFFFFULL}};

struct Run {
    const TargetSet* ts=nullptr;
    int mode=MD_RANDOM;
    std::atomic<u64> seq;                                // compteur partagé (modes séquentiel, Sobol, Pi, e, ln)
    int bits=0,nthreads=1,bsize=0,nbl=0;                 // nthreads = ncpu + ngpu (travailleurs)
    int ncpu=1,ngpu=0;                                   // cœurs CPU / GPU utilisés
    std::vector<int> gpu_ids; std::vector<std::string> gpu_names; int gpu_nthr=0;   // gpu_nthr>0 : nombre de threads GPU imposé (tests)
    U256 start,end,blockmask;
    u64 NB=1, NBP=1;                                     // NB = nombre de blocs ; NBP = puissance de 2 >= NB (suites Sobol, Pi, ...)
    u64 bwidth=0;                                        // 0 = blocs de 2^bsize clés ; sinon largeur 2R+1 (option -v)
    bool r_adjusted=false, r_fail=false;                 // R augmenté d'office (trop de blocs) / R impossible
    bool is_range=false; long double total=1;            // plage personnalisée ; nombre total de clés
    std::vector<std::atomic<u64>> claim;                // blocs déjà pris par un travailleur
    std::vector<std::atomic<u64>> dbits;                // blocs entièrement terminés (point de reprise)
    struct Item { u64 blk,off; };
    std::vector<Item> resume_q; SpinLock qlock;          // blocs en cours repris depuis un point de sauvegarde
    std::vector<TState> st;
    std::vector<GpuTrack> gtrack;
    std::atomic<bool> stop; std::atomic<int> done; std::atomic<int> nfound; std::atomic<int> gpu_lost;
    SpinLock lock; std::vector<Found> found;
    bool write_file=false, emergency=true, quiet=false;
    std::string out_main="cle_trouvee.txt", out_copy="cle_trouvee_copie.txt";   // deux copies du résultat
    std::string ckpt_path; double ckpt_every=120; u64 keys_base=0;               // point de reprise
    double time_limit=0; bool timed_out=false; double status_every=15;
    clk::time_point t0;
    Run():seq(0),stop(false),done(0),nfound(0),gpu_lost(0){}
};
static void setup_workers(Run& r,int ncpu,const std::vector<int>& gpus,const std::vector<std::string>* names=nullptr){
    r.ncpu=ncpu; r.ngpu=(int)gpus.size(); r.nthreads=ncpu+r.ngpu;
    r.gpu_ids=gpus; r.gpu_names.assign(gpus.size(),"GPU");
    if(names) for(size_t i=0;i<gpus.size()&&i<names->size();i++) r.gpu_names[i]=(*names)[i];
    r.st=std::vector<TState>(r.nthreads);
    r.gtrack=std::vector<GpuTrack>(r.ngpu);
    r.found.clear(); r.stop=false; r.done=0; r.nfound=0; r.gpu_lost=0;
}

static U256 block_start(const Run& r,u64 blk){
    U256 s=r.start;
    if(r.bwidth){ u128 v=(u128)blk*r.bwidth; U256 t={{(u64)v,(u64)(v>>64),0,0}}; add_raw(s,s,t); }
    else add_raw(s,s,shl_u64(blk,r.bsize));
    return s;
}
static U256 block_end(const Run& r,const U256& s){          // fin du bloc commençant en s (bornée par la fin de la plage)
    U256 e;
    if(r.bwidth){ U256 t={{r.bwidth-1,0,0,0}}; add_raw(e,s,t); } else add_raw(e,s,r.blockmask);
    if(!geq(r.end,e)) e=r.end;
    return e;
}
static std::string blk_txt(const Run& r){
    if(r.bwidth) return fmt_int(r.bwidth)+" (±"+fmt_int((r.bwidth-1)/2)+")";
    return "2^"+std::to_string(r.bsize);
}
// Plage quelconque [start,end] (1 <= start <= end < n). R > 0 : fenêtres de 2R+1 clés (voisinage)
static void init_run_range(Run& r,const U256& start,const U256& end,int nthreads,const TargetSet* ts,bool write_file,int mode=MD_RANDOM,bool is_range=true,u64 R=0){
    r.ts=ts; r.write_file=write_file; r.mode=mode; r.seq.store(0);
    r.start=start; r.end=end; r.is_range=is_range; r.bits=highest_bit(end)+1;
    U256 sz; sub_raw(sz,end,start);                     // taille - 1
    r.total=u256_to_ld(sz)+1;
    int L=highest_bit(sz)+1; if(L<0) L=0;               // taille <= 2^L
    // Taille des blocs (indépendante du matériel : le point de reprise reste valable d'une machine à l'autre) :
    // viser ~2^16 blocs, avec des blocs de 2^10 à 2^20 clés (un lot CPU = 1024 clés) (au plus 2^26 blocs)
    int b=L-16; if(b<10) b=std::min(10,std::max(0,L-6)); if(b>20) b=20;
    if(L-b>26) b=L-26;                                  // au plus 2^26 blocs
    r.bsize=b; r.blockmask=low_mask(b);
    U256 t=sz; for(int i=0;i<b;i++){ for(int k=0;k<4;k++){ t.d[k]>>=1; if(k<3) t.d[k]|=t.d[k+1]<<63; } }
    r.NB=t.d[0]+1;                                      // = ceil(taille / 2^b)  (<= 2^26)
    r.bwidth=0; r.r_adjusted=false; r.r_fail=false;
    if(R>0){                                            // voisinage : blocs de largeur W = 2R+1, ordonnés par la suite choisie
        u64 W=(R>(~0ULL-1)/2)? ~0ULL : 2*R+1;
        U256 q=div_small(sz,W);
        if(q.d[1]|q.d[2]|q.d[3]|(u64)(q.d[0]>=(1ULL<<26))){        // trop de blocs : on agrandit la fenêtre
            U256 w2=shr_n(sz,26); add_small(w2,1);
            if(w2.d[1]|w2.d[2]|w2.d[3]) r.r_fail=true;
            else { W=w2.d[0]; q=div_small(sz,W); r.r_adjusted=true; }
        }
        if(!r.r_fail){ r.bwidth=W; r.bsize=0; r.blockmask=ZERO; r.NB=q.d[0]+1; }
    }
    r.nbl=0; while((1ULL<<r.nbl)<r.NB) r.nbl++;
    r.NBP=1ULL<<r.nbl;
    u64 nw=(r.NBP+63)/64;
    r.claim=std::vector<std::atomic<u64>>(nw);
    r.dbits=std::vector<std::atomic<u64>>(nw);
    for(u64 i=0;i<nw;i++){ r.claim[i].store(0); r.dbits[i].store(0); }
    for(u64 bb=r.NB;bb<nw*64;bb++) r.claim[bb>>6].fetch_or(1ULL<<(bb&63));   // blocs hors plage = déjà pris
    r.resume_q.clear(); r.keys_base=0;
    setup_workers(r,nthreads,std::vector<int>());
}
// Plage d'un nombre de bits : [2^(bits-1), 2^bits - 1]
static void init_run(Run& r,int bits,int nthreads,const TargetSet* ts,bool write_file,int mode=MD_RANDOM,u64 R=0){
    U256 st=ZERO; st.d[(bits-1)>>6]|=1ULL<<((bits-1)&63);
    init_run_range(r,st,low_mask(bits),nthreads,ts,write_file,mode,false,R);
}
static inline void mark_done(Run& r,u64 blk){ r.dbits[blk>>6].fetch_or(1ULL<<(blk&63)); }


static inline bool claim_block(Run& r,u64 b){
    std::atomic<u64>& w=r.claim[b>>6]; u64 m=1ULL<<(b&63);
    if(w.load(std::memory_order_relaxed)&m) return false;
    return !(w.fetch_or(m)&m);
}
static inline u64 rev_bits(u64 v,int nb){
    u64 r=0; for(int i=0;i<nb;i++){ r=(r<<1)|(v&1); v>>=1; } return r;
}

// Générateurs d'ordre de blocs. Chaque bloc n'est donné qu'une seule fois (liste partagée "claim").
struct Gen {
    Run* r; int type; Rng rng;
    u64 ZS=1,hits=0; size_t pos=0;
    std::vector<u32> order;
    Gen(Run* rr,int t,int tid):r(rr),type(t){
        rng.seed((u64)clk::now().time_since_epoch().count()*0x9E3779B97F4A7C15ULL+(u64)tid*7919+(u64)t*104729);
        if(type==MD_MC){
            u64 NBp=r->NBP, Z=std::min<u64>(NBp,4096); ZS=NBp/Z;
            order.resize(Z); std::iota(order.begin(),order.end(),0u);
            for(size_t i=order.size();i>1;--i) std::swap(order[i-1],order[rng.next()%i]);
        }
    }
    // Balayage final : prend un bloc encore libre (garantit que toute la plage est couverte, sans bloc refait)
    bool sweep(u64& blk,u64 w0){
        u64 nw=r->claim.size();
        for(u64 j=0;j<nw;j++){
            u64 w=(w0+j)%nw; u64 cur=r->claim[w].load();
            while(~cur){
                u64 b=w*64+(u64)__builtin_ctzll(~cur);
                if(claim_block(*r,b)){ blk=b; return true; }
                cur=r->claim[w].load();
            }
        }
        return false;
    }
    bool next(u64& blk){
        u64 NB=r->NB;
        switch(type){
        case MD_SEQ:                                       // du début à la fin de la plage
            for(;;){
                u64 n=r->seq.fetch_add(1); if(n>=NB) return false;
                if(claim_block(*r,n)){ blk=n; return true; }
            }
        case MD_SOBOL:                                     // suite de Sobol (dimension 1, ordre de Gray) : permutation des blocs
            for(;;){
                u64 n=r->seq.fetch_add(1); if(n>=r->NBP) return false;
                u64 b=rev_bits(n^(n>>1),r->nbl);
                if(claim_block(*r,b)){ blk=b; return true; }
            }
        case MD_PI: case MD_E: case MD_LN: {                 // suite de Weyl : bloc = frac(i * c) * NB, puis balayage final
            const u64 C=WEYL_C[type-MD_PI], limit=4*r->NBP+64;
            for(;;){
                u64 i=r->seq.fetch_add(1);
                if(i>=limit) return sweep(blk,0);
                u64 b=r->nbl? ((i*C)>>(64-r->nbl)) : 0;
                if(claim_block(*r,b)){ blk=b; return true; }
            }
        }
        case MD_RANDOM:                                    // blocs tirés au hasard, sans répétition
            for(int t=0;t<16;t++){ u64 b=rng.next()&(r->NBP-1); if(claim_block(*r,b)){ blk=b; return true; } }
            return sweep(blk,rng.next()%r->claim.size());
        default: break;
        }
        for(;;){                                          // Monte Carlo : zones égales visitées dans un ordre aléatoire
            if(pos>=order.size()){
                if(pos>0&&hits==0) return false;
                for(size_t i=order.size();i>1;--i) std::swap(order[i-1],order[rng.next()%i]);
                pos=0; hits=0;
            }
            u64 base=(u64)order[pos++]*ZS;
            for(int t=0;t<8;t++){ u64 b=base+(rng.next()&(ZS-1)); if(claim_block(*r,b)){ hits++; blk=b; return true; } }
            u64 off=rng.next()&(ZS-1);
            for(u64 j=0;j<ZS;j++){ u64 b=base+((off+j)&(ZS-1)); if(claim_block(*r,b)){ hits++; blk=b; return true; } }
        }
    }
};


// ---------------------------------------------------------------- hôte : tables, périphériques, lanceur
static bool g_tab_ready=false;
static void gpu_prepare_tables(){
    if(g_tab_ready) return;
    build_tables();
    for(int i=0;i<=GHALF;i++){
        FE a=gf_from_u256(i==0?ZERO:gx[i]), b=gf_from_u256(i==0?ZERO:gy[i]);
        memcpy(h_tab.gx[i],a.v,32); memcpy(h_tab.gy[i],b.v,32);
    }
    U256 k2={{(u64)(2*GHALF),0,0,0}}; Aff s=aff_mul(k2,GEN);
    FE sx=gf_from_u256(s.x), sy=gf_from_u256(s.y), g1=gf_from_u256(GEN.x), g2=gf_from_u256(GEN.y);
    memcpy(h_tab.sst[0],sx.v,32); memcpy(h_tab.sst[1],sy.v,32);
    memcpy(h_tab.gen[0],g1.v,32); memcpy(h_tab.gen[1],g2.v,32);
    g_tab_ready=true;
}
#if REAL_CUDA
static bool cu_ok(cudaError_t e,const char* what,std::string& err){
    if(e==cudaSuccess) return true;
    err=std::string(what)+" : "+cudaGetErrorString(e); return false;
}
#endif
template<class T> static bool d_alloc(T*& p,size_t n,std::string& err){
    if(n==0) n=1;
#if REAL_CUDA
    void* q=nullptr;
    if(!cu_ok(cudaMalloc(&q,n*sizeof(T)),"cudaMalloc",err)) return false;
    if(!cu_ok(cudaMemset(q,0,n*sizeof(T)),"cudaMemset",err)){ cudaFree(q); return false; }
    p=(T*)q; return true;
#else
    p=(T*)calloc(n,sizeof(T)); if(!p){ err="mémoire insuffisante"; return false; } return true;
#endif
}
template<class T> static void d_free(T*& p){
    if(!p) return;
#if REAL_CUDA
    cudaFree((void*)p);
#else
    free((void*)p);
#endif
    p=nullptr;
}
static bool d_up(void* d,const void* h,size_t n,std::string& err){
#if REAL_CUDA
    return cu_ok(cudaMemcpy(d,h,n,cudaMemcpyHostToDevice),"cudaMemcpy (hôte->GPU)",err);
#else
    (void)err; memcpy(d,h,n); return true;
#endif
}
static bool d_down(void* h,const void* d,size_t n,std::string& err){
#if REAL_CUDA
    return cu_ok(cudaMemcpy(h,d,n,cudaMemcpyDeviceToHost),"cudaMemcpy (GPU->hôte)",err);
#else
    (void)err; memcpy(h,d,n); return true;
#endif
}

struct GpuInfo { int id=0; std::string name; int sm=0; size_t mem=0; int major=0,minor=0; };
static std::vector<GpuInfo> gpu_list(){
    std::vector<GpuInfo> v;
#if REAL_CUDA
    int n=0;
    if(cudaGetDeviceCount(&n)!=cudaSuccess){ cudaGetLastError(); return v; }
    for(int i=0;i<n;i++){
        cudaDeviceProp p;
        if(cudaGetDeviceProperties(&p,i)!=cudaSuccess){ cudaGetLastError(); continue; }
        GpuInfo g; g.id=i; g.name=p.name; g.sm=p.multiProcessorCount; g.mem=p.totalGlobalMem; g.major=p.major; g.minor=p.minor;
        v.push_back(g);
    }
#endif
    return v;
}

struct GpuRunner {
    int dev=0,nthr=0,nblk=0,tpb=GTPB,sm=1; std::string name="CPU (émulation)"; size_t memb=0;
    GArgs A; bool addr=false; unsigned long long kprev=0;
    u32 *d_st=nullptr,*d_cenok=nullptr,*d_asg=nullptr,*d_nhit=nullptr;
    u64 *d_lo=nullptr,*d_hi=nullptr,*d_cen=nullptr,*d_nlo=nullptr,*d_nhi=nullptr,*d_bloom=nullptr;
    GHit* d_hits=nullptr; unsigned long long* d_kdone=nullptr;
    std::vector<u64> h_nlo,h_nhi,h_lo; std::vector<u32> h_asg,h_st; std::vector<GHit> h_hits;
#if REAL_CUDA
    cudaEvent_t ev=nullptr;
#endif
    GpuRunner(){ memset(&A,0,sizeof A); }
    ~GpuRunner(){ release(); }
    GpuRunner(const GpuRunner&)=delete; GpuRunner& operator=(const GpuRunner&)=delete;
    void release(){
#if REAL_CUDA
        cudaSetDevice(dev);
        if(ev){ cudaEventDestroy(ev); ev=nullptr; }
#endif
        d_free(d_st); d_free(d_cenok); d_free(d_asg); d_free(d_nhit); d_free(d_lo); d_free(d_hi); d_free(d_cen);
        d_free(d_nlo); d_free(d_nhi); d_free(d_bloom); d_free(d_hits); d_free(d_kdone);
    }
    bool init(int d,const TargetSet& ts,int nthr_req,std::string& err){
        dev=d; addr=ts.addr;
        gpu_prepare_tables();
#if REAL_CUDA
        if(!cu_ok(cudaSetDevice(d),"cudaSetDevice",err)) return false;
        cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync); cudaGetLastError();     // le thread hôte dort pendant le calcul
        cudaDeviceProp pr; if(!cu_ok(cudaGetDeviceProperties(&pr,d),"cudaGetDeviceProperties",err)) return false;
        name=pr.name; sm=pr.multiProcessorCount; memb=pr.totalGlobalMem;
        if(!cu_ok(cudaEventCreateWithFlags(&ev,cudaEventDisableTiming|cudaEventBlockingSync),"cudaEventCreate",err)) return false;
        if(!cu_ok(cudaMemcpyToSymbol(c_tab,&h_tab,sizeof(GTables)),"cudaMemcpyToSymbol",err)) return false;
#endif
        tpb=GTPB;
        nblk=(nthr_req>0)? (nthr_req+tpb-1)/tpb : sm*GBLK_PER_SM;
        if(nblk<1) nblk=1;
        nthr=nblk*tpb;
        const size_t T=(size_t)nthr;
        if(!(d_alloc(d_st,T,err)&&d_alloc(d_cenok,T,err)&&d_alloc(d_asg,T,err)&&d_alloc(d_lo,T*4,err)&&d_alloc(d_hi,T*4,err)
             &&d_alloc(d_cen,T*8,err)&&d_alloc(d_nlo,T*4,err)&&d_alloc(d_nhi,T*4,err)&&d_alloc(d_hits,(size_t)GMAXHIT,err)
             &&d_alloc(d_nhit,1,err)&&d_alloc(d_kdone,1,err)&&d_alloc(d_bloom,ts.bloom.size(),err))) return false;
        if(!d_up(d_bloom,ts.bloom.data(),ts.bloom.size()*8,err)) return false;
        h_nlo.assign(T*4,0); h_nhi.assign(T*4,0); h_asg.assign(T,0); h_st.assign(T,0); h_lo.assign(T*4,0);
        A.st=d_st; A.lo=d_lo; A.hi=d_hi; A.cen=d_cen; A.cenok=d_cenok; A.asg=d_asg; A.nlo=d_nlo; A.nhi=d_nhi;
        A.bloom=d_bloom; A.bmask=ts.nblocks-1; A.hits=d_hits; A.nhit=d_nhit; A.maxhit=GMAXHIT; A.guard=GMAXHIT/2;
        A.kdone=d_kdone; A.nthr=nthr; A.budget=1;
        return true;
    }
    void set_assign(int t,const U256& lo,const U256& hi){
        for(int k=0;k<4;k++){ h_nlo[(size_t)t*4+k]=lo.d[k]; h_nhi[(size_t)t*4+k]=hi.d[k]; }
        h_asg[t]=1;
    }
    bool upload_assign(std::string& err){
        bool ok=d_up(d_nlo,h_nlo.data(),h_nlo.size()*8,err)&&d_up(d_nhi,h_nhi.data(),h_nhi.size()*8,err)&&d_up(d_asg,h_asg.data(),h_asg.size()*4,err);
        std::fill(h_asg.begin(),h_asg.end(),0u);
        return ok;
    }
    template<bool ADDR> void launch(){
#if defined(__CUDACC__) && !defined(CUDA_STUB)
        g_kernel<ADDR><<<nblk,tpb>>>(A);
#elif defined(__CUDACC__)
        for(int b=0;b<nblk;b++) for(int t=0;t<tpb;t++){ blockIdx.x=b; threadIdx.x=t; blockDim.x=tpb; g_kernel<ADDR>(A); }
#else
        for(int t=0;t<nthr;t++) g_thread<ADDR>(t,A);
#endif
    }
    bool run(int budget,std::string& err){
        A.budget=budget;
        if(addr) launch<true>(); else launch<false>();
#if REAL_CUDA
        if(!cu_ok(cudaGetLastError(),"lancement du noyau",err)) return false;
        if(!cu_ok(cudaEventRecord(ev,0),"cudaEventRecord",err)) return false;
        if(!cu_ok(cudaEventSynchronize(ev),"exécution du noyau",err)) return false;
#else
        (void)err;
#endif
        return true;
    }
    // Récupère l'état : h_st, h_lo, h_hits ; kdelta = clés testées depuis le dernier appel ; nhit_total = candidats émis
    bool fetch(unsigned long long& kdelta,u32& nhit_total,std::string& err){
        unsigned long long kd=0; u32 nh=0;
        if(!(d_down(h_st.data(),d_st,h_st.size()*4,err)&&d_down(h_lo.data(),d_lo,h_lo.size()*8,err)
             &&d_down(&kd,d_kdone,8,err)&&d_down(&nh,d_nhit,4,err))) return false;
        kdelta=kd-kprev; kprev=kd; nhit_total=nh;
        const u32 n=std::min<u32>(nh,GMAXHIT);
        h_hits.resize(n);
        if(n&&!d_down(h_hits.data(),d_hits,(size_t)n*sizeof(GHit),err)) return false;
        const u32 z=0;
        return d_up(d_nhit,&z,4,err);
    }
};

// ---------------------------------------------------------------- résultat trouvé
static Ent ent_of_point(bool addr,const Aff& P){
    Ent e;
    if(addr){ e.x=hash160_point_bytes(P); e.par=0; }
    else { e.x=P.x; e.par=P.y.d[0]&1; }
    return e;
}
// ---------------------------------------------------------------- enregistrement fiable des résultats
static std::string result_text(const Run& r,const Found& f){
    std::string t; char b[96];
    if(r.ts->addr){
        t+="Adresse (P2PKH)     : "+ent_addr_legacy(f.ent)+"\n";
        t+="Adresse (bech32)    : "+ent_addr_bech32(f.ent)+"\n";
        t+="Hash160             : "+ent_hash_hex(f.ent)+"\n";
    }
    t+="Clé publique        : "+f.pub+"\n";
    if(r.is_range) t+="Plage               : "+to_hex(r.start)+" -> "+to_hex(r.end)+"\n";
    else { snprintf(b,sizeof b,"Bits                : %d\n",r.bits); t+=b; }
    if(f.outside) t+="Hors plage demandée : OUI (clé valide, trouvée juste après la fin de la plage)\n";
    t+="Clé privée (hex)    : "+to_hex(f.key)+"\n";
    t+="Clé privée (décimal): "+to_dec(f.key)+"\n";
    t+="Vérification        : VALIDE\n";
    snprintf(b,sizeof b,"Temps (s)           : %.1f\n",f.t); t+=b;
    t+="----------------------------------------\n";
    return t;
}
// Ajout en fin de fichier, vidé sur le disque (flush + fsync)
static bool append_result(const std::string& path,const std::string& text){
    FILE* f=fopen(path.c_str(),"ab"); if(!f) return false;
    bool ok=fwrite(text.data(),1,text.size(),f)==text.size();
    if(fflush(f)!=0) ok=false;
#ifdef _WIN32
    if(_commit(_fileno(f))!=0) ok=false;
#else
    if(fsync(fileno(f))!=0) ok=false;
#endif
    if(fclose(f)!=0) ok=false;
    return ok;
}
// Écrit puis RELIT le fichier : le texte doit s'y trouver à la fin, sinon l'enregistrement est considéré comme raté
static bool save_checked(const std::string& path,const std::string& text){
    if(!append_result(path,text)) return false;
    std::string c; if(!read_file(path,c)) return false;
    return c.size()>=text.size() && c.compare(c.size()-text.size(),text.size(),text)==0;
}
static void record_find(Run& r,const U256& key,const Aff& P){
    r.lock.lock();
    for(const Found& f:r.found) if(eq(f.key,key)){ r.lock.unlock(); return; }
    Found f; f.key=key; f.ent=ent_of_point(r.ts->addr,P); f.pub=compress_hex(P);
    f.t=std::chrono::duration<double>(clk::now()-r.t0).count();
    f.outside=!geq(r.end,key)||!geq(key,r.start);
    f.saved=!r.write_file;
    if(r.write_file){
        std::string text=result_text(r,f), used; int ok=0;
        const std::string* names[2]={&r.out_main,&r.out_copy};
        for(int i=0;i<2;i++){
            if(names[i]->empty()) continue;
            if(save_checked(*names[i],text)){ ok++; if(!used.empty()) used+=" + "; used+=*names[i]; }
        }
        if(!ok&&r.emergency){                                  // secours : autres emplacements
            std::vector<std::string> alt;
            alt.push_back("cle_trouvee_"+std::to_string((long long)time(nullptr))+".txt");
            for(const char* ev:{"TEMP","TMP","TMPDIR","USERPROFILE","HOME"}) if(const char* d=getenv(ev)) alt.push_back(std::string(d)+"/cle_trouvee_urgence.txt");
            alt.push_back("/tmp/cle_trouvee_urgence.txt");
            for(const std::string& a2:alt) if(save_checked(a2,text)){ ok++; used=a2; break; }
        }
        f.saved=ok>0; f.path=used;
        if(!f.saved&&!r.quiet){
            fprintf(stderr,"\n\n*** ATTENTION : clé trouvée mais IMPOSSIBLE à enregistrer ! Note-la maintenant : ***\n%s\n",text.c_str());
            fflush(stderr);
        }
    }
    r.found.push_back(f);
    r.nfound.fetch_add(1);
    if(r.found.size()>=r.ts->list.size()) r.stop.store(true);
    r.lock.unlock();
}
// Confirmation indépendante : k*G recalculé en arithmétique affine (et hash160 par le chemin "octets"),
// puis comparaison exacte avec la liste des cibles.
static void verify_point(Run& r,const U256& k){
    if(is_zero(k)) return;
    Aff P=aff_mul(k,GEN);
    if(P.inf) return;
    bool hit=r.ts->addr ? ts_has_exact(*r.ts,hash160_point_bytes(P),0)
                        : ts_has_exact(*r.ts,P.x,P.y.d[0]&1);
    if(hit) record_find(r,k,P);
}

struct Scratch {
    std::vector<U256> pre,cx; std::vector<char> zero; std::vector<u64> cpar; std::vector<int> coff;
    Scratch():pre(HALF+2),cx(2*HALF+2*VLANES+2),zero(HALF+2),cpar(2*HALF+2*VLANES+2),coff(2*HALF+2*VLANES+2){}
};

// Parcourt [start,end] (un bloc) par lots de 2*HALF clés consécutives.
// Un point central C sert aux clés kc-i et kc+i : une seule inversion (par lots) pour deux clés.
template<bool ADDR>
static bool search_block(Run& r,const U256& start,const U256& end,const U256& bs,Scratch& sc,TState& st){
    const TargetSet& ts=*r.ts;
    const u64* bl=ts.bloom.data(); const u64 mask=ts.nblocks-1;
    U256 lo=start, kc=start; add_small(kc,(u64)HALF);          // clés du lot : kc-HALF .. kc+HALF-1
    Aff C=jac_to_aff(jac_mul(kc,GEN));
    for(;;){
        if(r.stop.load(std::memory_order_relaxed)) return false;
        U256 diff; sub_raw(diff,end,lo);
        bool last=((diff.d[1]|diff.d[2]|diff.d[3])==0) && diff.d[0]<(u64)(2*HALF);
        u64 count=last? diff.d[0]+1 : (u64)(2*HALF);
        int nc=0;
        auto key_of=[&](int off){ U256 k=kc; if(off>=0) add_small(k,(u64)off); else { U256 m={{(u64)(-off),0,0,0}}; sub_raw(k,kc,m); } return k; };
        auto emit=[&](const U256& x,u64 par,int off){
            if(ADDR){ sc.cx[nc]=x; sc.cpar[nc]=par; sc.coff[nc]=off; nc++; }
            else if(bloom_test(bl,mask,x) && ts_has_x(ts,x)) verify_point(r,key_of(off));
        };
        emit(C.x,C.y.d[0]&1,0);                                   // clé centrale

        // inversion par lots de (gx[i]-C.x), i = 1..HALF, puis du pas vers le lot suivant
        U256 acc=ONE;
        for(int i=1;i<=HALF;i++){
            U256 d; fe_sub(d,gx[i],C.x);
            if(is_zero(d)){ sc.zero[i]=1; d=ONE; } else sc.zero[i]=0;
            fe_mul(acc,acc,d); sc.pre[i]=acc;
        }
        U256 dS; fe_sub(dS,SST.x,C.x);
        bool zs=is_zero(dS); if(zs) dS=ONE;
        fe_mul(acc,acc,dS);
        U256 inv=fe_inv(acc);
        Aff Cn; Cn.inf=false; Cn.x=ZERO; Cn.y=ZERO;
        {
            U256 invd; fe_mul(invd,inv,sc.pre[HALF]); fe_mul(inv,inv,dS);
            if(!zs&&!last){
                U256 t,l; fe_sub(t,SST.y,C.y); fe_mul(l,t,invd);
                fe_sqr(Cn.x,l); fe_sub(Cn.x,Cn.x,C.x); fe_sub(Cn.x,Cn.x,SST.x);
                fe_sub(t,C.x,Cn.x); fe_mul(Cn.y,l,t); fe_sub(Cn.y,Cn.y,C.y);
            }
        }
        for(int i=HALF;i>=1;--i){
            U256 invd;
            if(i>1){
                fe_mul(invd,inv,sc.pre[i-1]);
                U256 d; fe_sub(d,gx[i],C.x); if(sc.zero[i]) d=ONE;
                fe_mul(inv,inv,d);
            } else invd=inv;
            if(sc.zero[i]){                                       // cas dégénéré (C = ±i*G) : chemin lent sûr
                if(i<HALF) verify_point(r,key_of(i));
                verify_point(r,key_of(-i));
                continue;
            }
            U256 t,l,x3,y3;
            // clé kc-i : C + (-iG)
            fe_add(t,gy[i],C.y); fe_mul(l,t,invd);
            fe_sqr(x3,l); fe_sub(x3,x3,C.x); fe_sub(x3,x3,gx[i]);
            u64 par=0;
            if(ADDR){ fe_sub(t,x3,C.x); fe_mul(y3,l,t); fe_sub(y3,y3,C.y); par=y3.d[0]&1; }
            emit(x3,par,-i);
            if(i<HALF){                                           // clé kc+i : C + iG
                fe_sub(t,gy[i],C.y); fe_mul(l,t,invd);
                fe_sqr(x3,l); fe_sub(x3,x3,C.x); fe_sub(x3,x3,gx[i]);
                if(ADDR){ fe_sub(t,C.x,x3); fe_mul(y3,l,t); fe_sub(y3,y3,C.y); par=y3.d[0]&1; }
                emit(x3,par,i);
            }
        }
        if(ADDR){                                                 // hash160 par paquets de VLANES clés
            for(int c=0;c<nc;c+=VLANES){
                U256 xs[VLANES],hs[VLANES]; u64 ps[VLANES]; int m=std::min(VLANES,nc-c);
                for(int l=0;l<VLANES;l++){ int idx=c+(l<m?l:m-1); xs[l]=sc.cx[idx]; ps[l]=sc.cpar[idx]; }
                hash160_vec(xs,ps,hs);
                for(int l=0;l<m;l++)
                    if(bloom_test(bl,mask,hs[l]) && ts_has_x(ts,hs[l])) verify_point(r,key_of(sc.coff[c+l]));
            }
        }
        st.keys.fetch_add(count,std::memory_order_relaxed);
        if(last) return true;
        add_small(lo,(u64)(2*HALF)); add_small(kc,(u64)(2*HALF));
        if(zs) C=jac_to_aff(jac_mul(kc,GEN)); else C=Cn;
        {   U256 dd; sub_raw(dd,lo,bs); u64 off=(dd.d[1]|dd.d[2]|dd.d[3])? ~0ULL : dd.d[0];    // position pour le point de reprise
            st.lk.lock(); st.ioff=off; st.lk.unlock(); }
    }
}


// ---------------------------------------------------------------- distribution des blocs, travailleurs CPU et GPU
static bool prepare_block(const Run& r,u64 blk,u64 off,U256& s,U256& e,U256& bs){
    bs=block_start(r,blk); e=block_end(r,bs); s=bs; add_small(s,off);
    return geq(e,s);
}
// Prend d'abord les blocs repris d'un point de sauvegarde, puis demande un nouveau bloc à la suite choisie
static bool next_block(Run& r,Gen& g,u64& blk,u64& off){
    r.qlock.lock();
    if(!r.resume_q.empty()){ Run::Item it=r.resume_q.back(); r.resume_q.pop_back(); r.qlock.unlock(); blk=it.blk; off=it.off; return true; }
    r.qlock.unlock();
    off=0; return g.next(blk);
}
struct WArg { Run* r; int tid; };
static void worker(WArg* a){
    Run& r=*a->r; int tid=a->tid; TState& st=r.st[tid];
    Gen g(&r,r.mode,tid);                                                   // tous les cœurs : le même mode
    Scratch sc;
    while(!r.stop.load()){
        u64 blk,off;
        if(!next_block(r,g,blk,off)) break;                                  // tous les blocs sont pris
        st.mode.store(g.type); st.block.store(blk);
        U256 bs,s,e;
        if(!prepare_block(r,blk,off,s,e,bs)){ mark_done(r,blk); continue; }
        st.lk.lock(); st.iact=true; st.iblk=blk; st.ioff=off; st.lk.unlock();
        bool ok=r.ts->addr? search_block<true>(r,s,e,bs,sc,st) : search_block<false>(r,s,e,bs,sc,st);
        if(ok){ mark_done(r,blk); st.lk.lock(); st.iact=false; st.lk.unlock(); }
    }
    st.mode.store(-1);
    r.done.fetch_add(1);
}

// Candidats du GPU : vérification exacte côté hôte (liste triée, puis k*G recalculé en arithmétique affine)
static void gpu_process_hits(Run& r,const std::vector<GHit>& hits){
    for(const GHit& h:hits){
        U256 key; for(int i=0;i<4;i++) key.d[i]=h.key[i];
        if(h.par==2){ verify_point(r,key); continue; }
        U256 v; for(int i=0;i<4;i++) v.d[i]=h.val[i];
        if(ts_has_x(*r.ts,v)) verify_point(r,key);
    }
}
static void gpu_worker(WArg* a){
    Run& r=*a->r; const int tid=a->tid, gi=tid-r.ncpu; TState& st=r.st[tid]; GpuTrack& tr=r.gtrack[gi];
    GpuRunner g; std::string err;
    if(!g.init(r.gpu_ids[gi],*r.ts,r.gpu_nthr,err)){
        fprintf(stderr,"\nGPU %d inutilisable : %s\n",r.gpu_ids[gi],err.c_str());
        r.gpu_lost.fetch_add(1); st.mode.store(-1); r.done.fetch_add(1); return;
    }
    const int T=g.nthr;
    tr.lk.lock(); tr.blk.assign(T,0); tr.off.assign(T,0); tr.act.assign(T,0); tr.lk.unlock();
    tr.nthr.store(T);
    Gen gen(&r,r.mode,tid);
    st.mode.store(gen.type);
    std::vector<char> busy(T,0); std::vector<u64> bid(T,0),boff(T,0);
    int nbusy=0; bool exhausted=false, failed=false, warned=false; int budget=2;
    while(!r.stop.load()){
        std::vector<int> newly;
        if(!exhausted){
            for(int t=0;t<T;t++){
                if(busy[t]) continue;
                u64 blk,off;
                if(!next_block(r,gen,blk,off)){ exhausted=true; break; }
                U256 bs,s,e;
                if(!prepare_block(r,blk,off,s,e,bs)){ mark_done(r,blk); t--; continue; }
                g.set_assign(t,s,e); busy[t]=1; bid[t]=blk; boff[t]=off; nbusy++; newly.push_back(t);
                st.block.store(blk);
            }
        }
        if(!newly.empty()){
            tr.lk.lock(); for(int t:newly){ tr.act[t]=1; tr.blk[t]=bid[t]; tr.off[t]=boff[t]; } tr.lk.unlock();
            if(!g.upload_assign(err)){ failed=true; break; }
        }
        if(nbusy==0) break;
        tr.busy.store(nbusy);
        const auto tl=clk::now();
        if(!g.run(budget,err)){ failed=true; break; }
        const double dt=std::chrono::duration<double>(clk::now()-tl).count();
        unsigned long long kd=0; u32 nh=0;
        if(!g.fetch(kd,nh,err)){ failed=true; break; }
        st.keys.fetch_add(kd);
        if(nh>GMAXHIT&&!warned&&!r.quiet){ warned=true; fprintf(stderr,"\nATTENTION : tampon de candidats GPU saturé (filtre de Bloom trop dense) : des résultats peuvent être perdus.\n"); }
        gpu_process_hits(r,g.h_hits);
        tr.lk.lock();
        for(int t=0;t<T;t++){
            if(!busy[t]) continue;
            if(g.h_st[t]==0){ mark_done(r,bid[t]); busy[t]=0; nbusy--; tr.act[t]=0; }
            else{
                U256 lo; for(int k=0;k<4;k++) lo.d[k]=g.h_lo[(size_t)t*4+k];
                U256 bs=block_start(r,bid[t]), dd; sub_raw(dd,lo,bs);
                tr.off[t]=(dd.d[1]|dd.d[2]|dd.d[3])? ~0ULL : dd.d[0];
            }
        }
        tr.lk.unlock();
        tr.busy.store(nbusy);
        // durée d'un lancement visée : ~0,25 s (assez court pour réagir à Ctrl-C, assez long pour être efficace)
        double f=(dt<0.01)? 4.0 : 0.25/dt; if(f>4) f=4; if(f<0.5) f=0.5;
        long long nb=(long long)(budget*f+0.5); if(nb<1) nb=1; if(nb>(1<<22)) nb=1<<22; budget=(int)nb;
    }
    if(failed){
        fprintf(stderr,"\nGPU %d arrêté : %s\n",r.gpu_ids[gi],err.c_str());
        r.gpu_lost.fetch_add(1);
    }
    tr.busy.store(0);
    st.mode.store(-1);
    r.done.fetch_add(1);
}

// ---------------------------------------------------------------- point de reprise (checkpoint)
static const char CK_MAGIC[8]={'P','Z','C','K','P','T','0','1'};
static u64 run_signature(const Run& r){
    u64 h=0xC0FFEE1234567ULL;
    h=sum_hash(r.ts->list.data(),r.ts->list.size()*sizeof(Ent),h);
    h=sum_hash(&r.start,sizeof(U256),h); h=sum_hash(&r.end,sizeof(U256),h);
    u64 ex[4]={r.bwidth,(u64)r.bsize,(u64)(r.ts->addr?1:0),r.NB};
    return sum_hash(ex,sizeof ex,h);
}
static std::string ckpt_default_path(const Run& r,const std::string& dir){
    char b[64]; snprintf(b,sizeof b,"puzzle_%016llx.ckpt",(unsigned long long)run_signature(r));
    return dir+b;
}
static bool save_ckpt(Run& r,const std::string& path,u64 keys_total){
    std::vector<Run::Item> infl;
    for(int i=0;i<r.ncpu;i++){ TState& s=r.st[i]; s.lk.lock(); if(s.iact){ Run::Item it={s.iblk,s.ioff}; infl.push_back(it); } s.lk.unlock(); }
    for(int g=0;g<r.ngpu;g++){
        GpuTrack& gt=r.gtrack[g]; gt.lk.lock();
        for(size_t t=0;t<gt.act.size();t++) if(gt.act[t]){ Run::Item it={gt.blk[t],gt.off[t]}; infl.push_back(it); }
        gt.lk.unlock();
    }
    std::vector<u64> words(r.dbits.size());                                   // lu APRÈS les blocs en cours
    for(size_t i=0;i<words.size();i++) words[i]=r.dbits[i].load();
    std::vector<Run::Item> keep;
    for(const Run::Item& e:infl) if(e.blk<r.NB&&!((words[e.blk>>6]>>(e.blk&63))&1)) keep.push_back(e);
    std::sort(keep.begin(),keep.end(),[](const Run::Item& a,const Run::Item& b){ return a.blk<b.blk||(a.blk==b.blk&&a.off<b.off); });
    keep.erase(std::unique(keep.begin(),keep.end(),[](const Run::Item& a,const Run::Item& b){ return a.blk==b.blk; }),keep.end());
    u64 H[16]={ENDIAN_TAG,r.start.d[0],r.start.d[1],r.start.d[2],r.start.d[3],r.end.d[0],r.end.d[1],r.end.d[2],r.end.d[3],
               (u64)r.bsize,r.bwidth,r.NB,run_signature(r),(u64)words.size(),(u64)keep.size(),keys_total};
    u64 ck=sum_hash(words.data(),words.size()*8,0xABCDEF01ULL);
    ck=sum_hash(keep.data(),keep.size()*sizeof(Run::Item),ck);
    ck=sum_hash(H,sizeof H,ck);
    std::string tmp=path+".tmp";
    FILE* f=fopen(tmp.c_str(),"wb"); if(!f) return false;
    bool ok=fwrite(CK_MAGIC,1,8,f)==8&&fwrite(H,8,16,f)==16&&fwrite(&ck,8,1,f)==1
          &&fwrite(words.data(),8,words.size(),f)==words.size()
          &&(keep.empty()||fwrite(keep.data(),sizeof(Run::Item),keep.size(),f)==keep.size());
    if(fflush(f)!=0) ok=false;
    if(ok&&fsync(fileno(f))!=0) ok=false;
    if(fclose(f)!=0) ok=false;
    if(!ok){ remove(tmp.c_str()); return false; }
    if(rename(tmp.c_str(),path.c_str())!=0){ remove(tmp.c_str()); return false; }
    return true;
}
// 1 = reprise chargée ; 0 = aucun fichier ; -1 = fichier présent mais incompatible/corrompu (ignoré)
static int load_ckpt(Run& r,const std::string& path,std::string& msg){
    FILE* f=fopen(path.c_str(),"rb"); if(!f) return 0;
    char mg[8]; u64 H[16],ck;
    if(fread(mg,1,8,f)!=8||memcmp(mg,CK_MAGIC,8)!=0||fread(H,8,16,f)!=16||fread(&ck,8,1,f)!=1){ fclose(f); msg="format invalide"; return -1; }
    const u64 nwords=H[13], ninfl=H[14];
    if(H[0]!=ENDIAN_TAG||nwords!=r.dbits.size()||ninfl>(1ULL<<32)){ fclose(f); msg="en-tête incompatible"; return -1; }
    std::vector<u64> words(nwords); std::vector<Run::Item> infl(ninfl);
    bool ok=fread(words.data(),8,nwords,f)==nwords&&(ninfl==0||fread(infl.data(),sizeof(Run::Item),ninfl,f)==ninfl);
    char extra; if(ok&&fread(&extra,1,1,f)==1) ok=false;
    fclose(f);
    if(!ok){ msg="fichier tronqué"; return -1; }
    u64 c2=sum_hash(words.data(),words.size()*8,0xABCDEF01ULL);
    c2=sum_hash(infl.data(),infl.size()*sizeof(Run::Item),c2);
    c2=sum_hash(H,sizeof H,c2);
    if(c2!=ck){ msg="somme de contrôle invalide"; return -1; }
    if(H[1]!=r.start.d[0]||H[2]!=r.start.d[1]||H[3]!=r.start.d[2]||H[4]!=r.start.d[3]||H[5]!=r.end.d[0]||H[6]!=r.end.d[1]||H[7]!=r.end.d[2]||H[8]!=r.end.d[3]
       ||H[9]!=(u64)r.bsize||H[10]!=r.bwidth||H[11]!=r.NB||H[12]!=run_signature(r)){ msg="autre recherche (plage, cibles ou découpage différents)"; return -1; }
    for(u64 i=0;i<nwords;i++){ r.dbits[i].store(words[i]); r.claim[i].fetch_or(words[i]); }
    r.resume_q.clear();
    for(const Run::Item& it:infl){
        if(it.blk>=r.NB||((words[it.blk>>6]>>(it.blk&63))&1)) continue;
        U256 s,e,bs; Run::Item q=it;
        if(!prepare_block(r,it.blk,it.off,s,e,bs)) q.off=0;                    // position incohérente : on refait tout le bloc
        r.claim[it.blk>>6].fetch_or(1ULL<<(it.blk&63));
        r.resume_q.push_back(q);
    }
    r.keys_base=H[15];
    return 1;
}

// ---------------------------------------------------------------- affichage
static bool g_ansi=false;
static std::atomic<bool> g_sigint(false);
static void on_sigint(int){ g_sigint.store(true); }
static const char* CL(const char* c){ return g_ansi?c:""; }
static const int INNER=74;
static int vis_len(const std::string& s){
    int n=0;
    for(size_t i=0;i<s.size();){
        unsigned char c=s[i];
        if(c==0x1b){ while(i<s.size()&&s[i]!='m') i++; i++; continue; }
        if((c&0xC0)!=0x80) n++;
        i++;
    }
    return n;
}
static std::string rep(const char* s,int n){ std::string o; for(int i=0;i<n;i++) o+=s; return o; }
static std::string row(const std::string& c){
    std::string o=std::string(CL("\x1b[90m"))+"║"+CL("\x1b[0m")+" "+c;
    int v=vis_len(c); if(v<INNER) o.append(INNER-v,' ');
    o+=std::string(" ")+CL("\x1b[90m")+"║"+CL("\x1b[0m")+"\n";
    return o;
}
static std::string border(const char* l,const char* m,const char* rr){
    return std::string(CL("\x1b[90m"))+l+rep(m,INNER+2)+rr+CL("\x1b[0m")+"\n";
}
static std::string padr(const std::string& s,int w){
    std::string o=s; int v=vis_len(s); if(v<w) o.append(w-v,' '); return o;
}
static const char* MODE_COL[MD_COUNT]={"\x1b[36m","\x1b[35m","\x1b[34m","\x1b[32m","\x1b[33m","\x1b[95m","\x1b[96m"};


struct Snap {
    double el=0; u64 tested=0; double speed=0, gpu_speed=0, cpu_speed=0;
    std::vector<double> tsp; std::vector<Found> found;
};
static std::string build_frame(Run& r,const Snap& s){
    std::string f;
    const TargetSet& ts=*r.ts;
    long double total=r.total;
    long double all=(long double)s.tested+(long double)r.keys_base;
    long double pct=100.0L*all/total; if(pct>100) pct=100;
    double avg=s.tested/(s.el>0?s.el:1e-9);
    long double eta=(total-all)/(avg>0?avg:1);
    char b[256];

    f+=border("╔","═","╗");
    f+=row(std::string(CL("\x1b[1m\x1b[36m"))+"PUZZLE BITCOIN · secp256k1 · GPU + CPU"+CL("\x1b[0m"));
    f+=border("╠","═","╣");
    double mb=(ts.bloom.size()*8.0+ts.list.size()*sizeof(Ent))/1048576.0;
    snprintf(b,sizeof b,"Cibles  : %s %s  (filtre + liste : %.1f Mo)",fmt_int(ts.list.size()).c_str(),ts.addr?"adresse(s)":"clé(s) publique(s)",mb);
    f+=row(b);
    if(r.is_range) snprintf(b,sizeof b,"Taille  : %s clés   Blocs : %s × %s",fmt_keys(r.total).c_str(),fmt_int(r.NB).c_str(),blk_txt(r).c_str());
    else snprintf(b,sizeof b,"Bits    : %d   Blocs : %s × %s clés",r.bits,fmt_int(r.NB).c_str(),blk_txt(r).c_str());
    f+=row(b);
    f+=row("Plage   : "+short_hex(r.start)+"  →  "+short_hex(r.end));
    f+=row(std::string("Mode    : ")+MODE_NAME[r.mode]+"  (tous les travailleurs sur la même suite de blocs)");
    snprintf(b,sizeof b,"Matériel: %d GPU + %d cœur(s) CPU%s",r.ngpu,r.ncpu,r.keys_base?"   (reprise d'un point de sauvegarde)":"");
    f+=row(b);
    f+=border("╠","═","╣");
    for(int g=0;g<r.ngpu&&g<8;g++){
        const int i=r.ncpu+g; const bool alive=r.st[i].mode.load()>=0;
        std::string nm=r.gpu_names[g]; if((int)nm.size()>18) nm=nm.substr(0,18);
        snprintf(b,sizeof b,"G%-2d ",g);
        std::string line=std::string(b)+CL("\x1b[96m")+padr(nm,18)+CL("\x1b[0m")+" ";
        snprintf(b,sizeof b,"%6d thr  %10.1f Mk/s%s",r.gtrack[g].nthr.load(),s.tsp[i]/1e6,alive?"":"  (arrêté)");
        f+=row(line+b);
    }
    int shown=std::min(r.ncpu,4);
    for(int i=0;i<shown;i++){
        int m=r.st[i].mode.load(); u64 blk=r.st[i].block.load();
        std::string mn=(m>=0)?MODE_NAME[m]:"terminé";
        std::string col=(m>=0&&g_ansi)?MODE_COL[m]:"";
        U256 bs=block_start(r,blk);
        snprintf(b,sizeof b,"C%-2d ",i+1);
        std::string line=b+col+padr(mn,12)+CL("\x1b[0m")+" ";
        snprintf(b,sizeof b,"bloc %9s  ",m>=0?fmt_int(blk).c_str():"-");
        line+=b;
        line+=padr(m>=0?short_hex(bs):"-",24);
        snprintf(b,sizeof b,"%6.2f Mk/s",s.tsp[i]/1e6);
        line+=b;
        f+=row(line);
    }
    if(r.ncpu>shown){ snprintf(b,sizeof b,"… +%d autres cœurs CPU   (CPU total : %.2f Mk/s)",r.ncpu-shown,s.cpu_speed/1e6); f+=row(CL("\x1b[90m")+std::string(b)+CL("\x1b[0m")); }
    f+=border("╠","═","╣");
    int W=40; int fl=(int)(pct/100.0L*W+0.0001L); if(fl>W) fl=W;
    snprintf(b,sizeof b,"  %6.2Lf %%",pct);
    f+=row(std::string(CL("\x1b[32m"))+rep("█",fl)+CL("\x1b[90m")+rep("░",W-fl)+CL("\x1b[0m")+b);
    snprintf(b,sizeof b,"Testé   : %-12s Vitesse : %.2f Mk/s",fmt_keys((long double)all).c_str(),s.speed/1e6);
    f+=row(b);
    snprintf(b,sizeof b,"Écoulé  : %-12s Reste   : ~%s",fmt_time(s.el).c_str(),fmt_time(eta).c_str());
    f+=row(b);
    snprintf(b,sizeof b,"Trouvées: %zu / %s",s.found.size(),fmt_int(ts.list.size()).c_str());
    f+=row(std::string(s.found.empty()?"":CL("\x1b[1m\x1b[32m"))+b+CL("\x1b[0m"));
    f+=border("╠","═","╣");
    int fstart=std::max(0,(int)s.found.size()-3);
    for(int i=0;i<3;i++){
        int idx=fstart+i;
        if(idx>=0&&idx<(int)s.found.size()){
            const Found& fd=s.found[idx];
            std::string pk=ent_text(ts,fd.ent); if(fd.outside) pk+=" (hors plage)"; pk=pk.substr(0,10)+"…"+pk.substr(pk.size()-6);
            std::string mark=fd.saved ? std::string(CL("\x1b[32m"))+"  ✔ enregistrée"+CL("\x1b[0m")
                                      : std::string(CL("\x1b[1m\x1b[31m"))+"  ⚠ NON ENREGISTRÉE"+CL("\x1b[0m");
            f+=row(std::string(CL("\x1b[32m"))+"★ "+CL("\x1b[0m")+pk+"  →  "+short_hex(fd.key)+mark);
        } else f+=row(i==0&&s.found.empty()?std::string(CL("\x1b[90m"))+"(aucune clé trouvée pour l'instant)"+CL("\x1b[0m"):"");
    }
    f+=border("╚","═","╝");
    return f;
}

static void run_engine(Run& r,bool display){
    r.t0=clk::now();
    const int n=r.nthreads;
    std::vector<WArg> args(n);
    for(int i=0;i<n;i++){ args[i].r=&r; args[i].tid=i; }
    std::vector<std::thread> ths;
    for(int i=0;i<n;i++){
        if(i<r.ncpu) ths.emplace_back(worker,&args[i]); else ths.emplace_back(gpu_worker,&args[i]);
    }
    std::vector<u64> prevk(n,0); std::vector<double> ema(n,0.0); double prevt=0; int prevLines=0; size_t shown=0; int lastnf=0;
    double last_status=-1e9, last_ck=0; bool ck_warned=false;
    auto session_keys=[&]()->u64{ u64 k=0; for(int i=0;i<n;i++) k+=r.st[i].keys.load(); return k; };
    auto draw=[&](bool final_){
        double el=std::chrono::duration<double>(clk::now()-r.t0).count();
        Snap s; s.el=el; s.tsp.assign(n,0.0);
        double dt=el-prevt; if(dt<=0) dt=1e-9;
        for(int i=0;i<n;i++){
            u64 k=r.st[i].keys.load(); double inst=(k-prevk[i])/dt; prevk[i]=k; s.tested+=k;
            ema[i]=(prevt==0)? inst : 0.6*ema[i]+0.4*inst;                    // lissage : les GPU ne publient qu'à chaque lancement
            s.tsp[i]=ema[i];
            if(i<r.ncpu) s.cpu_speed+=ema[i]; else s.gpu_speed+=ema[i];
        }
        prevt=el; s.speed=s.tested/(el>0?el:1e-9);
        r.lock.lock(); s.found=r.found; r.lock.unlock();
        if(!g_ansi){                                           // sans couleurs : annonce immédiate de chaque clé trouvée
            for(size_t i=shown;i<s.found.size();i++)
                printf("\n*** CLÉ TROUVÉE ***  clé privée : %s\n    %s\n",to_hex(s.found[i].key).c_str(),
                       s.found[i].saved?("enregistrée dans "+s.found[i].path).c_str():"!!! NON ENREGISTRÉE !!!");
            if(final_||el-last_status>=r.status_every){
                last_status=el;
                long double all=(long double)s.tested+(long double)r.keys_base;
                printf("[%6.2Lf%%] %.1f Mk/s (GPU %.1f | CPU %.2f) | écoulé %s | trouvées %zu/%zu\n",
                    100.0L*all/r.total,s.speed/1e6,s.gpu_speed/1e6,s.cpu_speed/1e6,fmt_time(el).c_str(),s.found.size(),r.ts->list.size());
            }
            fflush(stdout);
        } else {
            std::string f=build_frame(r,s);
            std::string out;
            if(s.found.size()>shown) out+="\a";                // signal sonore à chaque nouvelle clé
            if(prevLines>0){ char c[16]; snprintf(c,sizeof c,"\x1b[%dA",prevLines); out+=c; }
            out+=f; fputs(out.c_str(),stdout); fflush(stdout);
            prevLines=(int)std::count(f.begin(),f.end(),'\n');
        }
        shown=s.found.size();
    };
    if(display&&g_ansi){ fputs("\x1b[?25l",stdout); }
    if(display) draw(false);
    auto tp=clk::now();
    while(r.done.load()<n){
        std::this_thread::sleep_for(std::chrono::milliseconds(display?100:2));
        const double el=std::chrono::duration<double>(clk::now()-r.t0).count();
        if(g_sigint.load()) r.stop.store(true);
        if(r.time_limit>0&&el>=r.time_limit&&!r.stop.load()){ r.timed_out=true; r.stop.store(true); }
        int nf=r.nfound.load();
        if(display&&(nf!=lastnf||std::chrono::duration<double>(clk::now()-tp).count()>=(g_ansi?0.5:1.0))){ lastnf=nf; tp=clk::now(); draw(false); }
        if(!r.ckpt_path.empty()&&el-last_ck>=r.ckpt_every){
            last_ck=el;
            if(!save_ckpt(r,r.ckpt_path,r.keys_base+session_keys())&&!ck_warned){ ck_warned=true; fprintf(stderr,"\nAttention : impossible d'écrire le point de reprise %s\n",r.ckpt_path.c_str()); }
        }
    }
    for(auto& t:ths) t.join();
    if(!r.ckpt_path.empty()) save_ckpt(r,r.ckpt_path,r.keys_base+session_keys());
    if(display){
        draw(true);
        if(g_ansi) fputs("\x1b[?25h",stdout); else fputs("\n",stdout);
        fflush(stdout);
    }
}

static void init_curve(){
    parse_hex(GX_HEX,GEN.x); parse_hex(GY_HEX,GEN.y);
    GEN.inf=false;
}

static bool engine_test(int bits,const std::vector<u64>& keys,const std::vector<u64>& expect,int nthreads,bool addr=false,int mode=MD_RANDOM){
    TargetSet ts; ts.addr=addr;
    for(u64 k:keys){ U256 kk={{k,0,0,0}}; Aff P=aff_mul(kk,GEN); ts.list.push_back(ent_of_point(addr,P)); }
    finalize_targets(ts);
    Run r; init_run(r,bits,nthreads,&ts,false,mode);
    run_engine(r,false);
    if(r.found.size()!=expect.size()) return false;
    for(u64 k:expect){
        bool ok=false;
        for(const Found& f:r.found) if(f.key.d[0]==k&&!f.key.d[1]&&!f.key.d[2]&&!f.key.d[3]) ok=true;
        if(!ok) return false;
    }
    return true;
}


static bool range_engine_test(u64 a,u64 b,const std::vector<u64>& keys,const std::vector<u64>& expect,int nthreads,bool addr,int mode,u64 R=0){
    TargetSet ts; ts.addr=addr;
    for(u64 k:keys){ U256 kk={{k,0,0,0}}; ts.list.push_back(ent_of_point(addr,aff_mul(kk,GEN))); }
    finalize_targets(ts);
    Run r; U256 sa={{a,0,0,0}}, sb={{b,0,0,0}}; init_run_range(r,sa,sb,nthreads,&ts,false,mode,true,R);
    run_engine(r,false);
    if(r.found.size()!=expect.size()) return false;
    for(u64 k:expect){ bool ok=false; for(const Found& f:r.found) if(f.key.d[0]==k&&!f.key.d[1]&&!f.key.d[2]&&!f.key.d[3]) ok=true; if(!ok) return false; }
    return true;
}
static bool range_coverage_test(u64 a,u64 b,int mode,int ncores,u64 R=0){
    TargetSet ts; { Aff P=aff_mul(ONE,GEN); ts.list.push_back(ent_of_point(false,P)); } finalize_targets(ts);
    Run r; U256 sa={{a,0,0,0}}, sb={{b,0,0,0}}; init_run_range(r,sa,sb,ncores,&ts,false,mode,true,R);
    std::vector<Gen> gens; for(int t=0;t<ncores;t++) gens.emplace_back(&r,mode,t);
    std::vector<char> seen(r.NB,0); u64 got=0; std::vector<char> alive(ncores,1); int live=ncores;
    for(int t=0;live>0;t=(t+1)%ncores){
        if(!alive[t]) continue;
        u64 blk; if(!gens[t].next(blk)){ alive[t]=0; live--; continue; }
        if(blk>=r.NB||seen[blk]) return false;
        seen[blk]=1; got++;
    }
    // le dernier bloc doit se terminer exactement à la fin de la plage
    U256 s2=block_start(r,r.NB-1); U256 e2=block_end(r,s2);
    bool exact=eq(e2,r.end) || !r.bwidth;                 // avec -v : le dernier bloc s'arrête pile à la fin
    return got==r.NB && exact && geq(r.end,s2);
}

// Chaque mode doit donner chaque bloc une fois et une seule (plusieurs "cœurs" simulés en alternance).
static bool coverage_test(int bits,int mode,int ncores){
    TargetSet ts; { U256 k=ONE; Aff P=aff_mul(k,GEN); ts.list.push_back(ent_of_point(false,P)); } finalize_targets(ts);
    Run r; init_run(r,bits,ncores,&ts,false,mode);
    std::vector<Gen> gens; for(int t=0;t<ncores;t++) gens.emplace_back(&r,mode,t);
    std::vector<char> seen(r.NB,0); u64 got=0; std::vector<char> alive(ncores,1); int live=ncores;
    for(int t=0;live>0;t=(t+1)%ncores){
        if(!alive[t]) continue;
        u64 b; if(!gens[t].next(b)){ alive[t]=0; live--; continue; }
        if(b>=r.NB||seen[b]) return false;
        seen[b]=1; got++;
    }
    return got==r.NB;
}

// Enregistrement : écriture + relecture, copie de secours, panne d'un fichier, panne totale (sans plantage)
static bool save_test(){
    bool ok=true;
    TargetSet ts; U256 k={{0xABCDE,0,0,0}}; ts.list.push_back(ent_of_point(false,aff_mul(k,GEN))); finalize_targets(ts);
    const char* fa="selftest_a.tmp"; const char* fb="selftest_b.tmp";
    remove(fa); remove(fb);
    {   // 1) deux fichiers valides : les deux contiennent la clé
        Run r; init_run(r,20,1,&ts,true,MD_SEQ); r.out_main=fa; r.out_copy=fb; r.emergency=false; r.quiet=true;
        run_engine(r,false);
        std::string ca,cb; read_file(fa,ca); read_file(fb,cb);
        std::string hex=to_hex(k);
        ok=ok&&r.found.size()==1&&r.found[0].saved&&ca.find(hex)!=std::string::npos&&cb.find(hex)!=std::string::npos;
    }
    remove(fa); remove(fb);
    {   // 2) fichier principal impossible : la copie sauve la clé
        Run r; init_run(r,20,1,&ts,true,MD_SEQ); r.out_main="/dossier_inexistant_xyz/a.tmp"; r.out_copy=fb; r.emergency=false; r.quiet=true;
        run_engine(r,false);
        std::string cb; read_file(fb,cb);
        ok=ok&&r.found.size()==1&&r.found[0].saved&&cb.find(to_hex(k))!=std::string::npos;
    }
    remove(fb);
    {   // 3) aucun fichier possible : signalé comme NON enregistré, sans plantage
        Run r; init_run(r,20,1,&ts,true,MD_SEQ); r.out_main="/dossier_inexistant_xyz/a.tmp"; r.out_copy="/dossier_inexistant_xyz/b.tmp"; r.emergency=false; r.quiet=true;
        run_engine(r,false);
        ok=ok&&r.found.size()==1&&!r.found[0].saved;
    }
    return ok;
}


// ---------------------------------------------------------------- auto-tests du moteur GPU
// 1) arithmétique "GPU" exécutée côté hôte, comparée au code CPU de référence ; 2) recherches complètes sur le(s) GPU
static bool gpu_math_test(){
    bool ok=true; gpu_prepare_tables();
    Rng g; g.seed(4242); int bad=0;
    for(int n=0;n<3000;n++){
        U256 a,b; for(int i=0;i<4;i++){ a.d[i]=g.next(); b.d[i]=g.next(); }
        if(n%5==1){ a=PRIME; a.d[0]-=g.next()%3000; }
        if(n%7==2){ b=PRIME; b.d[0]-=g.next()%3000; }
        if(n%11==3){ a=ZERO; a.d[0]=g.next()%5; }
        if(n%13==4){ b=ZERO; b.d[0]=g.next()%5; }
        if(geq(a,PRIME)) sub_raw(a,a,PRIME);
        if(geq(b,PRIME)) sub_raw(b,b,PRIME);
        FE fa=gf_from_u256(a), fb=gf_from_u256(b), fr; U256 e;
        gf_add(fr,fa,fb); fe_add(e,a,b); if(!eq(gf_to_u256(fr),e)) bad++;
        gf_sub(fr,fa,fb); fe_sub(e,a,b); if(!eq(gf_to_u256(fr),e)) bad++;
        gf_mul(fr,fa,fb); fe_mul(e,a,b); if(!eq(gf_to_u256(fr),e)) bad++;
        gf_sqr(fr,fa); fe_sqr(e,a); if(!eq(gf_to_u256(fr),e)) bad++;
        { FE t=fa; gf_mul(t,t,t); fe_sqr(e,a); if(!eq(gf_to_u256(t),e)) bad++; }          // alias r == a
        if(n%40==0&&!is_zero(a)){ gf_inv(fr,fa); if(!eq(gf_to_u256(fr),fe_inv(a))) bad++; }
    }
    if(bad){ fprintf(stderr,"AUTO-TEST ÉCHEC : arithmétique GPU (%d écarts)\n",bad); ok=false; }
    int bh=0;
    for(int n=0;n<300;n++){
        U256 x; for(int i=0;i<4;i++) x.d[i]=g.next(); u64 par=g.next()&1;
        FE fx=gf_from_u256(x); u32 h[5]; g_hash160(fx.v,(u32)par,h);
        U256 want=hash160_fast(x,par), got; got.d[0]=h[0]|((u64)h[1]<<32); got.d[1]=h[2]|((u64)h[3]<<32); got.d[2]=h[4]; got.d[3]=0;
        if(!eq(want,got)) bh++;
    }
    if(bh){ fprintf(stderr,"AUTO-TEST ÉCHEC : hash160 GPU (%d écarts)\n",bh); ok=false; }
    int bp=0;
    U256 ks[5]={{{1,0,0,0}},{{2,0,0,0}},{{0xDEADBEEFCAFEBABEULL,0x0123456789ABCDEFULL,0x1111222233334444ULL,0x0ABCDEF012345678ULL}},
                {{0xFFFFFFFFFFFFFFFFULL,0,0,0}},{{0xBFD25E8CD0364140ULL,0xBAAEDCE6AF48A03BULL,0xFFFFFFFFFFFFFFFEULL,0xFFFFFFFFFFFFFFFFULL}}};
    for(int i=0;i<5;i++){
        Aff p=aff_mul(ks[i],GEN); FE x,y; g_point_of_key(ks[i],x,y);
        if(!eq(gf_to_u256(x),p.x)||!eq(gf_to_u256(y),p.y)) bp++;
    }
    if(bp){ fprintf(stderr,"AUTO-TEST ÉCHEC : multiplication scalaire GPU\n"); ok=false; }
    return ok;
}
static bool gpu_run_test(u64 a,u64 b,const std::vector<u64>& keys,const std::vector<u64>& expect,bool addr,int mode,u64 R,
                         int ncpu,const std::vector<int>& gpus,const std::vector<std::string>& gnames,int gthr){
    TargetSet ts; ts.addr=addr;
    for(u64 k:keys){ U256 kk={{k,0,0,0}}; ts.list.push_back(ent_of_point(addr,aff_mul(kk,GEN))); }
    finalize_targets(ts);
    Run r; U256 sa={{a,0,0,0}}, sb={{b,0,0,0}}; init_run_range(r,sa,sb,0,&ts,false,mode,true,R);
    setup_workers(r,ncpu,gpus,&gnames); r.gpu_nthr=gthr; r.quiet=true;
    run_engine(r,false);
    if(r.gpu_lost.load()) return false;
    if(r.found.size()!=expect.size()) return false;
    for(u64 k:expect){ bool ok=false; for(const Found& f:r.found) if(f.key.d[0]==k&&!f.key.d[1]&&!f.key.d[2]&&!f.key.d[3]) ok=true; if(!ok) return false; }
    return true;
}
// Reprise : un bloc terminé est ignoré, un bloc en cours reprend à sa position, le reste est cherché normalement
static bool ckpt_test(const std::vector<int>& gpus,const std::vector<std::string>& gnames){
    const u64 A=1000,B=500000,C=900000,D=300000;
    TargetSet ts; for(u64 k:{A,B,C,D}){ U256 kk={{k,0,0,0}}; ts.list.push_back(ent_of_point(false,aff_mul(kk,GEN))); } finalize_targets(ts);
    U256 sa={{1,0,0,0}}, sb={{1ULL<<20,0,0,0}};
    const char* path="selftest.ckpt"; remove(path);
    Run r1; init_run_range(r1,sa,sb,2,&ts,false,MD_SEQ,true,0);
    const u64 ba=(A-1)>>r1.bsize, bb=(B-1)>>r1.bsize, bc=(C-1)>>r1.bsize;
    mark_done(r1,ba);
    r1.claim[ba>>6].fetch_or(1ULL<<(ba&63));
    r1.st[0].iact=true; r1.st[0].iblk=bb; r1.st[0].ioff=500;       // B est avant la position : ne doit PAS être retrouvée
    r1.st[1].iact=true; r1.st[1].iblk=bc; r1.st[1].ioff=50;        // C est après la position : doit être retrouvée
    if(!save_ckpt(r1,path,12345)) return false;
    Run r2; init_run_range(r2,sa,sb,2,&ts,false,MD_RANDOM,true,0);
    std::string msg; if(load_ckpt(r2,path,msg)!=1||r2.keys_base!=12345||r2.resume_q.size()!=2) { remove(path); return false; }
    setup_workers(r2,2,gpus,&gnames); r2.gpu_nthr=64; r2.quiet=true;
    run_engine(r2,false);
    remove(path);
    bool fC=false,fD=false,other=false;
    for(const Found& f:r2.found){ if(f.key.d[0]==C) fC=true; else if(f.key.d[0]==D) fD=true; else other=true; }
    // une reprise incompatible (autre plage) doit être refusée
    Run r3; U256 sb2={{1ULL<<21,0,0,0}}; init_run_range(r3,sa,sb2,1,&ts,false,MD_SEQ,true,0);
    save_ckpt(r1,path,1); std::string m3; int rc=load_ckpt(r3,path,m3); remove(path);
    return fC&&fD&&!other&&r2.found.size()==2&&rc==-1;
}
static bool gpu_selftests(const std::vector<int>& gpus,const std::vector<std::string>& gnames){
    bool ok=true;
    auto check=[&](bool c,const char* name){ if(!c){ fprintf(stderr,"AUTO-TEST ÉCHEC : %s\n",name); ok=false; } };
    if(!gpu_math_test()) ok=false;
#if REAL_CUDA
    std::vector<int> ids=gpus; std::vector<std::string> nm=gnames;
#else
    std::vector<int> ids(1,0); std::vector<std::string> nm(1,"émulation"); (void)gpus; (void)gnames;   // sans CUDA : le noyau tourne en série sur le CPU
#endif
    for(size_t gi=0;gi<ids.size();gi++){
        std::vector<int> one(1,ids[gi]); std::vector<std::string> on(1,nm[gi]); const int T=64;
        std::string tag=" (GPU "+std::to_string(ids[gi])+")";
        #define GCHK(cond,txt) check(cond,(std::string(txt)+tag).c_str())
        GCHK(gpu_run_test(1,1,{1},{1},false,MD_SEQ,0,0,one,on,T),"GPU clé 1");
        GCHK(gpu_run_test(2,3,{3},{3},true,MD_SEQ,0,0,one,on,T),"GPU adresse bits=2");
        GCHK(gpu_run_test(4,7,{4,7,5},{4,7,5},false,MD_SOBOL,0,0,one,on,T),"GPU petites plages");
        GCHK(gpu_run_test(512,1023,{700,600,1000,5},{700,600,1000},false,MD_RANDOM,0,0,one,on,T),"GPU bits=10 multi-cibles");
        GCHK(gpu_run_test(512,1023,{700,600,1000,5},{700,600,1000},true,MD_SEQ,0,0,one,on,T),"GPU bits=10 adresses");
        GCHK(gpu_run_test(8192,16383,{9000,12000,16383,8192,300},{9000,12000,16383,8192},false,MD_PI,0,0,one,on,T),"GPU bits=14");
        GCHK(gpu_run_test(8192,16383,{9000,12000,16383,8192,300},{9000,12000,16383,8192},true,MD_MC,0,0,one,on,T),"GPU bits=14 adresses");
        {   std::vector<u64> ek; for(u64 o:{0,1,GHALF-1,GHALF,GHALF+1,2*GHALF-1,2*GHALF,2*GHALF+1,3*GHALF,1000,2047}) ek.push_back(2048+o);
            GCHK(gpu_run_test(2048,4095,ek,ek,false,MD_SEQ,0,0,one,on,T),"GPU frontières de lots");
            GCHK(gpu_run_test(2048,4095,ek,ek,true,MD_E,0,0,one,on,T),"GPU frontières de lots (adresse)"); }
        GCHK(gpu_run_test(1000,7777,{1000,1001,4321,7777,2},{1000,1001,4321,7777},false,MD_LN,0,0,one,on,T),"GPU plage quelconque");
        GCHK(gpu_run_test(100000,300000,{100000,300000,177777,50000},{100000,300000,177777},true,MD_RANDOM,0,0,one,on,T),"GPU plage 100000:300000 (adresse)");
        GCHK(gpu_run_test(100000,300000,{100000,300000,177777,50000},{100000,300000,177777},false,MD_SOBOL,2500,0,one,on,T),"GPU voisinage -v");
        GCHK(gpu_run_test(1000,7777,{1000,1001,4321,7777,2},{1000,1001,4321,7777},true,MD_SEQ,100,0,one,on,T),"GPU voisinage -v (adresse)");
        GCHK(gpu_run_test(16384,32767,{12345},{},false,MD_SEQ,0,0,one,on,T),"GPU clé absente");
        GCHK(gpu_run_test(100000,300000,{100000,300000,177777,50000},{100000,300000,177777},false,MD_RANDOM,0,2,one,on,T),"GPU + 2 cœurs CPU");
        #undef GCHK
    }
    check(ckpt_test(ids,nm),"point de reprise (CPU + GPU)");
    return ok;
}

static bool self_test(const std::vector<int>& gpus,const std::vector<std::string>& gnames){
    bool ok=true;
    auto check=[&](bool c,const char* name){ if(!c){ fprintf(stderr,"AUTO-TEST ÉCHEC : %s\n",name); ok=false; } };
    check(on_curve(GEN),"G sur la courbe");
    U256 ex;
    Aff g2=aff_mul({{2,0,0,0}},GEN); parse_hex("C6047F9441ED7D6D3045406E95C07CD85C778E4B8CEF3CA7ABAC09B95C709EE5",ex);
    check(eq(g2.x,ex),"2G");
    Aff g3=aff_mul({{3,0,0,0}},GEN); parse_hex("F9308A019258C31049344F85F89D5229B531C845836F99B08601F113BCE036F9",ex);
    check(eq(g3.x,ex),"3G");
    U256 ks[4]={{{0xDEADBEEFCAFEBABEULL,0x0123456789ABCDEFULL,0x1111222233334444ULL,0x0ABCDEF012345678ULL}},
                {{0xFFFFFFFFFFFFFFFFULL,0,0,0}},{{0,1,0,0}},{{12345,0,0xABCDULL,0}}};
    for(int i=0;i<4;i++){
        Aff a=aff_mul(ks[i],GEN), j=jac_to_aff(jac_mul(ks[i],GEN));
        check(eq(a.x,j.x)&&eq(a.y,j.y)&&on_curve(a),"affine == jacobien");
    }
    build_tables();
    for(int i=1;i<=HALF;i++){
        Aff j=jac_to_aff(jac_mul({{(u64)i,0,0,0}},GEN));
        if(!(eq(j.x,gx[i])&&eq(j.y,gy[i]))){ check(false,"table i*G"); break; }
    }
    Aff ng=jac_to_aff(jac_mul({{(u64)(2*HALF),0,0,0}},GEN));
    check(eq(ng.x,SST.x)&&eq(ng.y,SST.y),"pas 2*HALF*G");

    // arithmétique : carré dédié, inversion par chaîne, hash160 par paquets
    {
        Rng g; g.seed(777); int bad=0;
        for(int n=0;n<4000;n++){
            U256 a; for(int i=0;i<4;i++) a.d[i]=g.next();
            if(n%5==1){ a=PRIME; a.d[0]-=g.next()%3000; }
            if(n%7==2){ a=ZERO; a.d[0]=g.next()%5; }
            if(geq(a,PRIME)) sub_raw(a,a,PRIME);
            U256 s1,s2; fe_sqr(s1,a); fe_mul(s2,a,a); if(!eq(s1,s2)) bad++;
            if(n%40==0&&!is_zero(a)){
                U256 inv=fe_inv(a),pr; fe_mul(pr,inv,a); if(!eq(pr,ONE)) bad++;
                U256 e=PRIME; e.d[0]-=2; if(!eq(inv,fe_pow(a,e))) bad++;
            }
        }
        check(bad==0,"carré / inversion");
        int bh=0;
        for(int n=0;n<200;n++){
            U256 xs[VLANES],hs[VLANES]; u64 ps[VLANES];
            for(int l=0;l<VLANES;l++){ for(int i=0;i<4;i++) xs[l].d[i]=g.next(); ps[l]=g.next()&1; }
            hash160_vec(xs,ps,hs);
            for(int l=0;l<VLANES;l++) if(!eq(hs[l],hash160_fast(xs[l],ps[l]))) bh++;
        }
        check(bh==0,"hash160 par paquets");
    }
    // SHA-256 / RIPEMD-160 / adresses (vecteurs connus : clé privée 1)
    {
        unsigned char pb[33],h[20]; pub33(GEN.x,GEN.y.d[0]&1,pb);
        hash160_bytes(pb,33,h);
        U256 want; parse_hex("00000000751e76e8199196d454941c45d1b3a323f1433bd6",want);
        unsigned char wb[20]={0x75,0x1e,0x76,0xe8,0x19,0x91,0x96,0xd4,0x54,0x94,0x1c,0x45,0xd1,0xb3,0xa3,0x23,0xf1,0x43,0x3b,0xd6};
        check(memcmp(h,wb,20)==0,"hash160(G) octets");
        check(eq(hash160_fast(GEN.x,GEN.y.d[0]&1),pack20(wb)),"hash160(G) rapide");
        check(eq(hash160_fast(g2.x,g2.y.d[0]&1),hash160_point_bytes(g2)),"hash160 rapide == octets (2G)");
        check(eq(hash160_fast(g3.x,g3.y.d[0]&1),hash160_point_bytes(g3)),"hash160 rapide == octets (3G)");
        (void)want;
        Ent e; e.x=pack20(wb); e.par=0;
        check(ent_addr_legacy(e)=="1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMH","encodage Base58Check");
        check(ent_addr_bech32(e)=="bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t4","encodage Bech32");
        Ent d1,d2,d3; 
        check(parse_address("1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMH",d1)==1&&eq(d1.x,e.x),"décodage Base58Check");
        check(parse_address("BC1QW508D6QEJXTDG4Y5R3ZARVARY0C5XW7KV8F3T4",d2)==1&&eq(d2.x,e.x),"décodage Bech32 (majuscules)");
        check(parse_address("1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMI",d3)==0,"checksum Base58 invalide");
        check(parse_address("bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t5",d3)==0,"checksum Bech32 invalide");
        check(parse_address("3J98t1WpEZ73CNmQviecrnyiWrnqRhWNLy",d3)==2,"P2SH non supporté");
        check(parse_address("bc1p0xlxvlhemja6c4dqv22uapctqupfhlxm9h8z3k2e72q4k9hcz7vqzk5jj0",d3)==2,"Taproot non supporté");
    }
    // lecture de texte : doublons, lignes invalides, CRLF, majuscules
    {
        Aff p1=aff_mul({{5,0,0,0}},GEN), p2=aff_mul({{77,0,0,0}},GEN);
        std::string a=compress_hex(p1), b=compress_hex(p2), A=a;
        for(auto& c:A) if(c>='a'&&c<='f') c-=32;
        std::string txt=a+"\r\n\r\n"+A+"\n"+b+" # commentaire\nzzzz\n"+std::string("04")+a.substr(2)+"\n"+a.substr(0,66-1)+"\n";
        TargetSet ts; u64 l,rj,dp,un; parse_targets_text(txt,ts,false,l,rj,dp,un);
        check(ts.list.size()==2&&dp==1&&rj==3&&l==6,"lecture du fichier texte");
        // aller-retour du fichier .bloom
        ts.src_hash=12345;
        const char* tmp="selftest.tmp.bloom";
        if(save_bloom(ts,tmp)){
            TargetSet t2; std::string err;
            bool lo=load_bloom(tmp,t2,err);
            check(lo&&t2.list.size()==2&&t2.src_hash==12345&&t2.nblocks==ts.nblocks,"relecture .bloom");
            if(lo) check(ts_has_exact(t2,p1.x,p1.y.d[0]&1)&&ts_has_exact(t2,p2.x,p2.y.d[0]&1)&&!ts_has_x(t2,g2.x),"contenu .bloom");
            // corruption volontaire : doit être détectée
            FILE* f=fopen(tmp,"r+b");
            if(f){ fseek(f,-3,SEEK_END); int c=fgetc(f); fseek(f,-3,SEEK_END); fputc(c^0x55,f); fclose(f);
                   TargetSet t3; std::string e2; check(!load_bloom(tmp,t3,e2),"détection de corruption"); }
            remove(tmp);
        }
    }
    // recherches complètes sur des clés connues (cas dégénérés, multi-cibles, multi-threads)
    check(engine_test(1,{1},{1},1),"bits=1");
    check(engine_test(2,{3},{3},1),"bits=2");
    check(engine_test(3,{4,7,5},{4,7,5},2),"bits=3");
    check(engine_test(7,{100},{100},1),"bits=7");
    check(engine_test(10,{700,600,1000,5},{700,600,1000},1),"bits=10 multi-cibles (1 thread)");
    check(engine_test(10,{700,600,1000,5},{700,600,1000},4),"bits=10 multi-cibles (4 threads)");
    check(engine_test(11,{1031,2047},{1031,2047},3),"bits=11");
    check(engine_test(17,{70000,66000,131071},{70000,66000,131071},4),"bits=17");
    check(engine_test(20,{0xFFFFF,0x80000,0x9ABCD},{0xFFFFF,0x80000,0x9ABCD},4),"bits=20");
    check(engine_test(22,{3000001},{3000001},2),"bits=22");
    check(engine_test(16,{12345},{},1),"clé absente de la plage");
    check(save_test(),"enregistrement fiable des résultats");
    {   // plages quelconques (taille non puissance de 2)
        bool cv=true;
        for(int m=0;m<MD_COUNT;m++) for(int nc:{1,3})
            for(auto pr:std::vector<std::pair<u64,u64>>{{1,1},{5,5},{3,70},{1000,7777},{65536,65536+100000},{12345,12345+2000000},{77,(1ULL<<25)+99}})
                cv=cv&&range_coverage_test(pr.first,pr.second,m,nc);
        check(cv,"couverture exacte de plages quelconques");
        bool ok=true;
        for(int m=0;m<MD_COUNT;m++) for(bool ad:{false,true}){
            ok=ok&&range_engine_test(1000,7777,{1000,1001,4321,7777,2},{1000,1001,4321,7777},2,ad,m);      // 2 est hors plage
            ok=ok&&range_engine_test(5,5,{5},{5},1,ad,m);
            ok=ok&&range_engine_test(100000,300000,{100000,300000,177777,50000},{100000,300000,177777},3,ad,m);
        }
        check(ok,"recherche dans des plages quelconques");
        // voisinage (-v R) : fenêtres de 2R+1 clés, chaque clé couverte une fois, toutes les modes
        bool cv2=true, ok2=true;
        for(int m=0;m<MD_COUNT;m++) for(int nc:{1,3})
            for(u64 R:{1ULL,3ULL,500ULL,10000ULL})
                for(auto pr:std::vector<std::pair<u64,u64>>{{1,1},{5,5},{3,70},{1000,7777},{12345,12345+500000}})
                    cv2=cv2&&range_coverage_test(pr.first,pr.second,m,nc,R);
        check(cv2,"couverture exacte avec voisinage (-v)");
        for(int m=0;m<MD_COUNT;m++) for(bool ad:{false,true}){
            ok2=ok2&&range_engine_test(1000,7777,{1000,1001,4321,7777,2},{1000,1001,4321,7777},2,ad,m,100);
            ok2=ok2&&range_engine_test(100000,300000,{100000,300000,177777,50000},{100000,300000,177777},3,ad,m,2500);
        }
        check(ok2,"recherche avec voisinage (-v)");
        {   Run rj; TargetSet t1; { Aff P=aff_mul(ONE,GEN); t1.list.push_back(ent_of_point(false,P)); } finalize_targets(t1);
            U256 sa={{1,0,0,0}}, sb={{1ULL<<30,0,0,0}};
            init_run_range(rj,sa,sb,1,&t1,false,MD_SOBOL,true,1);
            check(rj.r_adjusted&&!rj.r_fail&&rj.NB<=(1ULL<<26)&&rj.bwidth>=3,"R trop petit : fenêtre agrandie automatiquement"); }
    }
    for(int m=0;m<MD_COUNT;m++){                     // chaque mode : couverture exacte des blocs + recherche réelle
        bool cv=true;
        for(int b:{1,2,5,7,10,14,19,27,33,40}) for(int nc:{1,3,4}) cv=cv&&coverage_test(b,m,nc);
        check(cv,(std::string("couverture des blocs, mode ")+MODE_NAME[m]).c_str());
        check(engine_test(14,{9000,12000,16383,8192},{9000,12000,16383,8192},3,false,m),"recherche, mode clé publique");
        check(engine_test(14,{9000,12000,16383,8192},{9000,12000,16383,8192},3,true,m),"recherche, mode adresse");
    }
    {   // clés aux frontières des lots de 2*HALF clés (début, milieu, fin, lot suivant)
        std::vector<u64> ek; for(u64 o:{0,1,HALF-1,HALF,HALF+1,2*HALF-1,2*HALF,2*HALF+1,3*HALF,1000,2047}) ek.push_back(2048+o);
        check(engine_test(12,ek,ek,1),"frontières de lots (clé publique)");
        check(engine_test(12,ek,ek,3,true),"frontières de lots (adresse)");
        std::vector<u64> e2; for(u64 o:{0,255,256,511,512,100000,262143}) e2.push_back(262144+o);
        check(engine_test(19,e2,e2,2),"frontières de lots, bloc plus grand");
        check(engine_test(19,e2,e2,2,true),"frontières de lots, bloc plus grand (adresse)");
    }
    check(engine_test(10,{700,600,1000,5},{700,600,1000},1,true),"adresses bits=10 (1 thread)");
    check(engine_test(14,{9000,12000,16383,300},{9000,12000,16383},4,true),"adresses bits=14 (4 threads)");
    check(engine_test(2,{3},{3},1,true),"adresses bits=2");
    if(!gpu_selftests(gpus,gnames)) ok=false;
    return ok;
}

// Diagnostic : si le programme s'arrête brutalement, on affiche pourquoi (et on rend le curseur).
static Run* g_run=nullptr;
static void dump_found_emergency(){
    Run* r=g_run; if(!r||r->found.empty()) return;
    fprintf(stderr,"Clés déjà trouvées (elles sont déjà enregistrées dans le fichier de résultats) :\n");
    for(const Found& f:r->found) fprintf(stderr,"  clé privée : %s   (clé publique %s)\n",to_hex(f.key).c_str(),f.pub.c_str());
    fflush(stderr);
}
static void on_terminate(){
    fputs("\x1b[?25h",stdout); fflush(stdout);
    fprintf(stderr,"\n\nERREUR : exception C++ non rattrapée, arrêt du programme.\n"); fflush(stderr);
    dump_found_emergency();
    std::_Exit(3);
}
static bool looks_like_pub(const std::string& s){
    return s.size()==66&&s[0]=='0'&&(s[1]=='2'||s[1]=='3');
}
static std::string lower(std::string s){ for(auto& c:s) if(c>='A'&&c<='Z') c+=32; return s; }
#include <sched.h>
// Nombre de cœurs réellement utilisables (affinité + quota cgroup des conteneurs Colab/Kaggle/Docker)
static int usable_cpus(){
    int n=(int)std::thread::hardware_concurrency(); if(n<1) n=1;
    cpu_set_t cs; CPU_ZERO(&cs);
    if(sched_getaffinity(0,sizeof cs,&cs)==0){ int c=CPU_COUNT(&cs); if(c>0&&c<n) n=c; }
    double q=-1;
    if(FILE* f=fopen("/sys/fs/cgroup/cpu.max","r")){
        char a[64]; long long p=0; if(fscanf(f,"%63s %lld",a,&p)==2&&strcmp(a,"max")!=0&&p>0) q=atof(a)/(double)p; fclose(f);
    } else if(FILE* f1=fopen("/sys/fs/cgroup/cpu/cpu.cfs_quota_us","r")){
        long long qu=-1,p=0; if(fscanf(f1,"%lld",&qu)==1&&qu>0){ if(FILE* f2=fopen("/sys/fs/cgroup/cpu/cpu.cfs_period_us","r")){ if(fscanf(f2,"%lld",&p)==1&&p>0) q=(double)qu/(double)p; fclose(f2); } } fclose(f1);
    }
    if(q>0){ int c=(int)std::ceil(q-1e-9); if(c<1) c=1; if(c<n) n=c; }
    return n<1?1:n;
}
static bool parse_duration(const std::string& s,double& out){
    if(s.empty()) return false;
    double tot=0; size_t i=0;
    while(i<s.size()){
        size_t j=i; while(j<s.size()&&(isdigit((unsigned char)s[j])||s[j]=='.')) j++;
        if(j==i) return false;
        double v=atof(s.substr(i,j-i).c_str()); double m=1;
        if(j<s.size()){ char u=s[j]; if(u=='h'||u=='H') m=3600; else if(u=='m'||u=='M') m=60; else if(u=='s'||u=='S') m=1; else return false; j++; }
        tot+=v*m; i=j;
    }
    out=tot; return tot>0;
}
static void usage(const char* prog){
    fprintf(stderr,
        "Usage : %s <cible> <bits | début:fin> [address] [mode] [options]\n"
        "  <bits | début:fin> : nombre de bits, ou plage de clés en hexadécimal (ex. 2000000:3ffffff)\n"
        "  <cible> : clé publique compressée, adresse, fichier .txt (une cible par ligne) ou fichier .bloom\n"
        "  address : à ajouter quand les cibles sont des adresses Bitcoin (1... ou bc1q...)\n"
        "  mode (un seul) : -r aléatoire (défaut) | -s séquentiel | -sobol | -pi | -e | -ln | -m Monte Carlo\n"
        "  -v R          : voisinage : chaque clé générée est le centre d'une fenêtre [k-R, k+R] (ex. -sobol -v 10000)\n"
        "  -t N          : cœurs CPU utilisés (défaut : tous ; 0 = GPU seuls)\n"
        "  --gpus L      : all (défaut) | none | indices séparés par des virgules (ex. 0,1)      (--no-gpu = none)\n"
        "  --out DIR     : dossier des résultats et du point de reprise\n"
        "  --ckpt FILE | --no-ckpt | --fresh | --ckpt-every S : point de reprise\n"
        "  --time T      : arrêt propre après T (ex. 3600, 90m, 11h30m)\n"
        "  --status S    : intervalle des lignes d'état hors terminal (défaut 15 s)     --skip-test : saute les auto-tests\n"
        "  --info | --selftest | --genpub <clé_hex> | --genaddr <clé_hex>\n",prog);
}
static void print_info(){
    printf("CPU : %d cœur(s) utilisables\n",usable_cpus());
    std::vector<GpuInfo> g=gpu_list();
#if REAL_CUDA
    if(g.empty()) printf("GPU : aucun GPU CUDA détecté (activer un accélérateur GPU dans Kaggle/Colab)\n");
    for(const GpuInfo& x:g) printf("GPU %d : %s  (%d SM, %.1f Go, calcul %d.%d)\n",x.id,x.name.c_str(),x.sm,x.mem/1073741824.0,x.major,x.minor);
#else
    printf("GPU : binaire compilé sans CUDA (CPU seul)\n");
#endif
}

int main(int argc,char** argv){
    g_ansi=isatty(1)||getenv("PUZZLE_ANSI");
    std::set_terminate(on_terminate);
    init_curve();
    if(argc==3 && (std::string(argv[1])=="--genpub"||std::string(argv[1])=="--genaddr")){
        U256 k; if(!parse_hex(argv[2],k)||is_zero(k)){ fprintf(stderr,"Clé invalide\n"); return 1; }
        Aff P=aff_mul(k,GEN);
        if(std::string(argv[1])=="--genpub") printf("%s\n",compress_hex(P).c_str());
        else { Ent e=ent_of_point(true,P); printf("%s\n%s\n",ent_addr_legacy(e).c_str(),ent_addr_bech32(e).c_str()); }
        return 0;
    }
    if(argc==2&&std::string(argv[1])=="--info"){ print_info(); return 0; }
    std::vector<GpuInfo> allgpu=gpu_list();
    if(argc==2&&std::string(argv[1])=="--selftest"){
        std::vector<int> ids; std::vector<std::string> nm;
        for(const GpuInfo& x:allgpu){ ids.push_back(x.id); nm.push_back(x.name); }
        printf("Auto-tests en cours... "); fflush(stdout);
        build_tables();
        bool ok=self_test(ids,nm); printf(ok?"OK\n":"ÉCHEC\n"); return ok?0:2;
    }
    if(argc<3){ usage(argv[0]); return 1; }
    std::string arg=argv[1];
    long bits=0; bool range_given=false; U256 rstart=ZERO, rend=ZERO;
    {
        std::string ra=argv[2]; size_t sep=ra.find_first_of(":-");
        if(sep!=std::string::npos){
            auto clean=[](std::string h){ if(h.size()>=2&&h[0]=='0'&&(h[1]=='x'||h[1]=='X')) h=h.substr(2); return h; };
            std::string h1=clean(ra.substr(0,sep)), h2=clean(ra.substr(sep+1));
            if(!parse_hex(h1,rstart)||!parse_hex(h2,rend)){ fprintf(stderr,"Plage invalide : écris-la en hexadécimal, par exemple 2000000:3ffffff\n"); return 1; }
            if(is_zero(rstart)){ fprintf(stderr,"Plage invalide : le début doit être au moins 1.\n"); return 1; }
            if(!geq(rend,rstart)){ fprintf(stderr,"Plage invalide : le début doit être inférieur ou égal à la fin.\n"); return 1; }
            if(geq(rend,CURVE_N)){ fprintf(stderr,"Plage invalide : la fin doit être inférieure à l'ordre de la courbe secp256k1.\n"); return 1; }
            range_given=true;
        } else {
            char* endp; bits=strtol(argv[2],&endp,10);
            if(*endp||bits<1||bits>255){ fprintf(stderr,"Nombre de bits invalide (1 à 255), ou plage début:fin en hexadécimal.\n"); return 1; }
        }
    }
    bool addrmode=false; int ncpu=usable_cpus(); int mode=MD_RANDOM; bool mode_set=false; u64 Rv=0;
    std::string gpusel="all", outdir, ckpt_user; bool no_ckpt=false, fresh=false, skip_test=false; double ckpt_every=120, time_limit=0, status_every=15;
    for(int i=3;i<argc;i++){
        std::string o=lower(argv[i]);
        int m=-1;
        auto need=[&](const char* nm)->const char*{ if(i+1>=argc){ fprintf(stderr,"%s doit être suivi d'une valeur.\n",nm); return nullptr; } return argv[++i]; };
        if(o=="address"){ addrmode=true; continue; }
        else if(o=="-r") m=MD_RANDOM; else if(o=="-s") m=MD_SEQ; else if(o=="-sobol") m=MD_SOBOL;
        else if(o=="-pi") m=MD_PI; else if(o=="-e") m=MD_E; else if(o=="-ln") m=MD_LN; else if(o=="-m") m=MD_MC;
        if(m>=0){
            if(mode_set){ fprintf(stderr,"Un seul mode à la fois (%s et %s indiqués).\n",MODE_NAME[mode],MODE_NAME[m]); usage(argv[0]); return 1; }
            mode=m; mode_set=true; continue;
        }
        if(o=="-v"){
            std::string a2=(i+1<argc)?argv[i+1]:"";
            if(a2.empty()||a2.size()>18||a2.find_first_not_of("0123456789")!=std::string::npos||strtoull(a2.c_str(),nullptr,10)==0){
                fprintf(stderr,"-v doit être suivi d'un rayon R entier > 0 (ex. -v 10000).\n"); return 1; }
            Rv=strtoull(a2.c_str(),nullptr,10); i++; continue;
        }
        if(o=="-t"){
            const char* v=need("-t"); if(!v) return 1;
            std::string sv=v; if(sv.empty()||sv.find_first_not_of("0123456789")!=std::string::npos){ fprintf(stderr,"-t doit être suivi d'un nombre de cœurs.\n"); return 1; }
            ncpu=atoi(v); continue;
        }
        if(o=="--gpus"){ const char* v=need("--gpus"); if(!v) return 1; gpusel=lower(v); continue; }
        if(o=="--no-gpu"){ gpusel="none"; continue; }
        if(o=="--out"){ const char* v=need("--out"); if(!v) return 1; outdir=v; continue; }
        if(o=="--ckpt"){ const char* v=need("--ckpt"); if(!v) return 1; ckpt_user=v; continue; }
        if(o=="--no-ckpt"){ no_ckpt=true; continue; }
        if(o=="--fresh"){ fresh=true; continue; }
        if(o=="--skip-test"){ skip_test=true; continue; }
        if(o=="--ckpt-every"){ const char* v=need("--ckpt-every"); if(!v) return 1; if(!parse_duration(v,ckpt_every)){ fprintf(stderr,"Durée invalide : %s\n",v); return 1; } continue; }
        if(o=="--status"){ const char* v=need("--status"); if(!v) return 1; if(!parse_duration(v,status_every)){ fprintf(stderr,"Durée invalide : %s\n",v); return 1; } continue; }
        if(o=="--time"){ const char* v=need("--time"); if(!v) return 1; if(!parse_duration(v,time_limit)){ fprintf(stderr,"Durée invalide : %s (ex. 3600, 90m, 11h30m)\n",v); return 1; } continue; }
        if(!o.empty()&&o.find_first_not_of("0123456789")==std::string::npos){ ncpu=atoi(o.c_str()); continue; }   // ancien format : nombre seul
        fprintf(stderr,"Argument inconnu : %s\n",argv[i]); usage(argv[0]); return 1;
    }
    if(ncpu>256) ncpu=256;

    // ---- choix des GPU
    std::vector<int> gids; std::vector<std::string> gnames;
    if(gpusel!="none"){
        if(gpusel=="all"){ for(const GpuInfo& x:allgpu){ gids.push_back(x.id); gnames.push_back(x.name); } }
        else{
            size_t p=0;
            while(p<=gpusel.size()){
                size_t q=gpusel.find(',',p); if(q==std::string::npos) q=gpusel.size();
                std::string tok=gpusel.substr(p,q-p); p=q+1;
                if(tok.empty()||tok.find_first_not_of("0123456789")!=std::string::npos){ fprintf(stderr,"--gpus : liste invalide (ex. all, none, 0,1).\n"); return 1; }
                int id=atoi(tok.c_str()); bool found_=false;
                for(const GpuInfo& x:allgpu) if(x.id==id){ found_=true; if(std::find(gids.begin(),gids.end(),id)==gids.end()){ gids.push_back(id); gnames.push_back(x.name); } }
                if(!found_){ fprintf(stderr,"GPU %d introuvable (voir --info).\n",id); return 1; }
            }
        }
    }
    if(ncpu<0) ncpu=0;
    if(ncpu==0&&gids.empty()){ fprintf(stderr,"Aucun travailleur : 0 cœur CPU et aucun GPU. Voir --info.\n"); return 1; }
    if(!outdir.empty()&&outdir.back()!='/') outdir+="/";
    if(!outdir.empty()){ std::error_code ec; std::filesystem::create_directories(outdir,ec); }

    // ---- chargement des cibles
    TargetSet ts; ts.addr=addrmode;
    Ent single;
    bool is_single_pub=!addrmode&&looks_like_pub(arg)&&parse_pub(arg,single);
    int addr_rc=addrmode?parse_address(arg,single):0;
    if(!addrmode&&looks_like_pub(arg)&&!is_single_pub){ fprintf(stderr,"Clé publique invalide ou hors de la courbe secp256k1.\n"); return 1; }
    if(is_single_pub||(addrmode&&addr_rc==1)){
        ts.list.push_back(single); finalize_targets(ts);
        if(addrmode) printf("Cible unique : %s  (hash160 %s)\n",ent_addr_legacy(single).c_str(),ent_hash_hex(single).c_str());
        else printf("Cible unique : %s\n",ent_hex(single).c_str());
    } else if(addrmode&&addr_rc==2){
        fprintf(stderr,"Adresse valide mais non supportée (seules les adresses P2PKH '1...' et P2WPKH 'bc1q...' le sont).\n"); return 1;
    } else {
        bool isbloom=arg.size()>=6&&arg.substr(arg.size()-6)==".bloom";
        if(isbloom){
            std::string err; TargetSet tmp;
            if(!load_bloom(arg,tmp,err)){ fprintf(stderr,"Impossible de charger %s : %s\n",arg.c_str(),err.c_str()); return 1; }
            if(tmp.addr!=addrmode){
                fprintf(stderr,tmp.addr?"Ce filtre contient des adresses : ajoute l'argument address.\n"
                                       :"Ce filtre contient des clés publiques : retire l'argument address.\n");
                return 1;
            }
            ts=tmp;
            printf("Filtre chargé : %s  (%s %s)\n",arg.c_str(),fmt_int(ts.list.size()).c_str(),ts.addr?"adresses":"clés publiques");
        } else {
            std::string txt;
            if(!read_file(arg,txt)){
                if(!addrmode&&(parse_address(arg,single)!=0))
                    fprintf(stderr,"Cette cible est une adresse Bitcoin : ajoute l'argument address.\n");
                else fprintf(stderr,"Impossible de lire %s (ni clé publique, ni adresse, ni fichier valide).\n",arg.c_str());
                return 1;
            }
            u64 h=fnv64(txt);
            size_t dot=arg.find_last_of('.'), slash=arg.find_last_of("/\\");
            std::string bname=((dot==std::string::npos||(slash!=std::string::npos&&dot<slash))?arg:arg.substr(0,dot))+".bloom";
            std::string err; TargetSet old;
            if(load_bloom(bname,old,err)&&old.src_hash==h&&old.addr==addrmode){
                ts=old; printf("Filtre existant réutilisé : %s  (%s cibles)\n",bname.c_str(),fmt_int(ts.list.size()).c_str());
            } else {
                u64 lines,rej,dups,unsup; parse_targets_text(txt,ts,addrmode,lines,rej,dups,unsup); ts.src_hash=h;
                if(ts.list.empty()){
                    fprintf(stderr,"Aucune cible valide dans %s (%s lignes lues).%s\n",arg.c_str(),fmt_int(lines).c_str(),
                        addrmode?"":" Si ce sont des adresses, ajoute l'argument address.");
                    return 1;
                }
                printf("Fichier lu : %s lignes, %s valides, %s doublons, %s rejetées",
                    fmt_int(lines).c_str(),fmt_int(ts.list.size()).c_str(),fmt_int(dups).c_str(),fmt_int(rej).c_str());
                if(unsup) printf(", %s non supportées (P2SH/P2WSH/Taproot)",fmt_int(unsup).c_str());
                printf("\n");
                if(save_bloom(ts,bname)) printf("Filtre enregistré : %s\n",bname.c_str());
                else fprintf(stderr,"Attention : impossible d'écrire %s\n",bname.c_str());
            }
        }
    }

    build_tables();
    if(!skip_test){
        printf("Auto-tests en cours... "); fflush(stdout);
        if(!self_test(gids,gnames)){ printf("ÉCHEC\n"); return 2; }
        printf("OK\n");
    }

    signal(SIGINT,on_sigint); signal(SIGTERM,on_sigint);
    Run run;
    if(range_given) init_run_range(run,rstart,rend,ncpu,&ts,true,mode,true,Rv); else init_run(run,(int)bits,ncpu,&ts,true,mode,Rv);
    if(run.r_fail){ fprintf(stderr,"Voisinage -v %llu impossible : plage trop grande pour des fenêtres aussi petites.\n",(unsigned long long)Rv); return 1; }
    setup_workers(run,ncpu,gids,&gnames);
    run.out_main=outdir+"cle_trouvee.txt"; run.out_copy=outdir+"cle_trouvee_copie.txt";
    run.time_limit=time_limit; run.status_every=status_every<1?1:status_every; run.ckpt_every=ckpt_every<5?5:ckpt_every;
    if(Rv){
        if(run.r_adjusted) printf("Voisinage : R=%s donnerait trop de blocs (> 2^26) : fenêtre portée à %s clés.\n",fmt_int(Rv).c_str(),fmt_int(run.bwidth).c_str());
        else printf("Voisinage : chaque clé générée couvre ±%s (fenêtre de %s clés, %s fenêtres).\n",fmt_int(Rv).c_str(),fmt_int(run.bwidth).c_str(),fmt_int(run.NB).c_str());
        if(run.bwidth<1024) printf("Attention : fenêtre < 1024 clés : le calcul par lots est sous-utilisé (plus lent). R >= 512 conseillé.\n");
    }
    printf("Matériel : %d GPU",(int)gids.size());
    for(size_t i=0;i<gids.size();i++) printf("%s%s",i?", ":" (",gnames[i].c_str());
    printf("%s + %d cœur(s) CPU\n",gids.empty()?"":")",ncpu);
    if(!no_ckpt){
        run.ckpt_path=ckpt_user.empty()? ckpt_default_path(run,outdir) : ckpt_user;
        if(!fresh){
            std::string msg; int rc=load_ckpt(run,run.ckpt_path,msg);
            if(rc==1){
                u64 nd=0; for(auto& w:run.dbits) nd+=(u64)__builtin_popcountll(w.load());
                printf("Reprise : %s blocs déjà terminés sur %s, %s bloc(s) en cours, %s clés déjà testées  (%s)\n",
                    fmt_int(nd).c_str(),fmt_int(run.NB).c_str(),fmt_int(run.resume_q.size()).c_str(),fmt_keys((long double)run.keys_base).c_str(),run.ckpt_path.c_str());
            } else if(rc<0) printf("Point de reprise ignoré (%s) : %s\n",run.ckpt_path.c_str(),msg.c_str());
        }
        printf("Point de reprise : %s (toutes les %.0f s)\n",run.ckpt_path.c_str(),run.ckpt_every);
    }
    g_run=&run;
    run_engine(run,true);

    // ---- récapitulatif
    double el=std::chrono::duration<double>(clk::now()-run.t0).count();
    bool interrupted=g_sigint.load();
    printf("\n");
    bool all_saved=true;
    for(const Found& f:run.found){
        if(ts.addr)
            printf("  ✔ %s  /  %s\n      clé publique : %s\n",ent_addr_legacy(f.ent).c_str(),ent_addr_bech32(f.ent).c_str(),f.pub.c_str());
        else printf("  ✔ %s\n",f.pub.c_str());
        printf("      clé privée   : %s\n      décimal      : %s\n",to_hex(f.key).c_str(),to_dec(f.key).c_str());
        if(f.outside) printf("      NB : clé valide, mais juste après la fin de la plage demandée\n");
        if(f.saved) printf("      enregistrée  : %s\n",f.path.c_str());
        else { all_saved=false; printf("      !!! NON ENREGISTRÉE : recopie cette clé tout de suite !!!\n"); }
    }
    printf("\n%zu / %zu cible(s) trouvée(s) en %s%s%s.\n",run.found.size(),ts.list.size(),fmt_time(el).c_str(),
           interrupted?" (interrompu)":"",run.timed_out?" (temps limite atteint)":"");
    if(run.gpu_lost.load()) printf("ATTENTION : %d GPU arrêté(s) en cours de route : la couverture de la plage est incomplète (blocs en cours conservés dans le point de reprise).\n",run.gpu_lost.load());
    if((interrupted||run.timed_out)&&!run.ckpt_path.empty()) printf("Point de reprise écrit : relance la même commande pour continuer.\n");
    if(!run.found.empty()&&all_saved) printf("Résultats enregistrés (et relus pour vérification) : %s\n",run.found.back().path.c_str());
    if(!all_saved){ fprintf(stderr,"\n*** ATTENTION : au moins une clé n'a pas pu être enregistrée dans un fichier ! ***\n"); return 4; }
    return run.found.size()==ts.list.size()?0:(interrupted?130:1);
}
