/* Weight distribution of the span of k vectors in F_3^n (n <= 128), by odometer enumeration.
   Input (stdin): n k, then k rows of n entries. Output: n+1 counts.
   Vectors are bitsliced: (a,b) with a = positions equal to 1, b = positions equal to 2.
   Incrementing the base-3 counter adds +1 to digits 0..j, i.e. adds prefix sum P_j. */
#include <stdio.h>
#include <stdint.h>
typedef unsigned __int128 u128;
static inline void add3(u128 xa,u128 xb,u128 ya,u128 yb,u128*ra,u128*rb){
    u128 t=(xa|yb)^(xb|ya);
    *ra=(xb|yb)^t; *rb=(xa|ya)^t;
}
static inline int pc(u128 x){return __builtin_popcountll((uint64_t)x)+__builtin_popcountll((uint64_t)(x>>64));}
int main(){
    int n,k; if(scanf("%d %d",&n,&k)!=2) return 1;
    u128 ga[64]={0},gb[64]={0},Pa[64],Pb[64];
    for(int i=0;i<k;i++) for(int l=0;l<n;l++){int v; if(scanf("%d",&v)!=1) return 1; v%=3; if(v<0)v+=3;
        if(v==1) ga[i]|=((u128)1)<<l; else if(v==2) gb[i]|=((u128)1)<<l;}
    u128 sa=0,sb=0;
    for(int j=0;j<k;j++){add3(sa,sb,ga[j],gb[j],&sa,&sb); Pa[j]=sa; Pb[j]=sb;}
    unsigned long long W[129]={0};
    unsigned char dig[64]={0};
    u128 xa=0,xb=0; W[0]++;
    for(;;){
        int j=0; while(j<k && dig[j]==2){dig[j]=0;j++;}
        if(j==k) break;               /* all digits wrapped: done */
        dig[j]++;
        add3(xa,xb,Pa[j],Pb[j],&xa,&xb);
        W[pc(xa|xb)]++;
    }
    for(int w=0;w<=n;w++) printf("%llu ",W[w]); printf("\n");
    return 0;
}
