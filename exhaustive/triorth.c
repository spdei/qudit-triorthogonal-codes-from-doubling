/*
 * triorth.c -- exhaustive search for short triorthogonal codes [[n,1,d>=3]]_p,
 * p in {3,5,7}: cubic conditions only (self-orthogonality not required), coefficients allowed.
 *
 * Reduced problem (the normal form of the paper):  A = [I_r | P] (stabilizers), h = (0 | q) (logical),
 * weights c_i on the identity columns and w_l on the P columns.
 *   (a) c_i + sum_l w_l P_il^3 = 0
 *   (b) sum_l w_l P_il^2 P_jl = 0                     (i != j)
 *   (c) sum_l w_l P_il P_jl P_kl = 0                  (i,j,k distinct)
 *   (d) lambda = sum_l w_l q_l^3 = 1                  (normalised)
 *   (e) sum_l w_l q_l^2 P_il = 0
 *   (f) sum_l w_l q_l P_il P_jl = 0                   (all i,j)
 *   (g) columns of P: >= 2 nonzeros, pairwise non-parallel   (<=> d >= 3)
 * Weights are classes of F_p^* modulo cubes: only for p = 7 is this nontrivial,
 * with representatives 1,2,3 (classes {1,6},{2,5},{3,4}).  In "unit" mode the
 * cost of a column of weight class k is k (number of unit-weight copies needed),
 * and the search is over total cost n; in "weighted" mode n = r + m.
 *
 * Symmetry breaking (all valid simultaneously, lex-min argument):
 *   columns of [w; q; P] lex non-decreasing; rows (c_i, p_i) lex non-decreasing;
 *   for p = 7: first nonzero entry of every column of [q;P] and of every row p_i
 *   lies in {1,3} (scaling by cube roots of unity {1,2,4}).
 *
 * Usage: ./triorth p n [-u] [-d D] [-f] [-v]
 *   -u   unit-weight cost mode (p = 7 only; p = 3,5 are automatically unit)
 *   -d D require d_Z >= D (checked by brute force at leaves; default 3)
 *   -f   stop at first code found
 *   -v   print every code found (default: first 3 per (r,m))
 *   -R   testing only: skip (g) (projectivity), to compare with brute force
 *   -rm R M  restrict to one pair (r, m)
 * Output: one line per (r, m) with search-tree nodes, number of codes and
 * CPU time; codes are printed as rows "w" (weights), "h", "a1".."ar", with
 * the identity block separated by a space.  "codes" counts double-lex
 * representatives, not equivalence classes.
 * Compile: gcc -O2 -march=native -o triorth triorth.c
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#define MAXM 32
#define MAXR 16
#define MAXL 256

static int p, r, m, dtarget = 3, unitmode = 0, stopfirst = 0, verbose = 0;
static int W[MAXM], C[MAXR], Q[MAXM], P[MAXR][MAXM];
static int cube[8], sq[8], inv[8], lead_ok[8];
static long long nodes, found, found_total;
static int done_flag;

static inline int md(int x) { x %= p; return x < 0 ? x + p : x; }

/* ------------------------------------------------------------------ */
static int relax = 0, fix_r = -1, fix_m = -1;
static int final_check(void)
{
    if (relax) return 1;
    /* (g): columns of P have >=2 nonzeros and are pairwise non-parallel */
    int norm[MAXM][MAXR];
    for (int l = 0; l < m; l++) {
        int nz = 0, f = 0;
        for (int i = 0; i < r; i++) if (P[i][l]) { if (!nz) f = P[i][l]; nz++; }
        if (nz < 2) return 0;
        int s = inv[f];
        for (int i = 0; i < r; i++) norm[l][i] = (P[i][l] * s) % p;
        for (int l2 = 0; l2 < l; l2++)
            if (!memcmp(norm[l], norm[l2], sizeof(int) * r)) return 0;
    }
    if (dtarget <= 3) return 1;
    /* brute-force: no v in A^perp \ H^perp of weight s, 3 <= s < dtarget */
    int n = r + m;
    static int col[MAXR + MAXM][MAXR + 1]; /* column l: (h_l, A_1l..A_rl) */
    for (int l = 0; l < n; l++) {
        if (l < r) { col[l][0] = 0; for (int i = 0; i < r; i++) col[l][i + 1] = (i == l); }
        else { col[l][0] = Q[l - r]; for (int i = 0; i < r; i++) col[l][i + 1] = P[i][l - r]; }
    }
    for (int s = 3; s < dtarget; s++) {
        int S[16]; for (int t = 0; t < s; t++) S[t] = t;
        while (1) {
            /* coefficients: first fixed to 1, others in 1..p-1 */
            int a[16]; a[0] = 1; for (int t = 1; t < s; t++) a[t] = 1;
            while (1) {
                int okA = 1;
                for (int i = 1; i <= r && okA; i++) {
                    int z = 0; for (int t = 0; t < s; t++) z += a[t] * col[S[t]][i];
                    if (z % p) okA = 0;
                }
                if (okA) {
                    int z = 0; for (int t = 0; t < s; t++) z += a[t] * col[S[t]][0];
                    if (z % p) return 0;
                }
                int t = 1; while (t < s && a[t] == p - 1) { a[t] = 1; t++; }
                if (t >= s) break;
                a[t]++;
            }
            int t = s - 1; while (t >= 0 && S[t] == n - s + t) t--;
            if (t < 0) break;
            S[t]++;
            for (int u = t + 1; u < s; u++) S[u] = S[u - 1] + 1;
        }
    }
    return 1;
}

static void report(void)
{
    found++; found_total++;
    if (verbose || found <= 3) {
        printf("  CODE p=%d r=%d m=%d\n", p, r, m);
        printf("    w : ");
        for (int i = 0; i < r; i++) printf("%d", C[i]);
        printf(" ");
        for (int l = 0; l < m; l++) printf("%d", W[l]);
        printf("\n");
        printf("    h : ");
        for (int i = 0; i < r; i++) printf("0");
        printf(" ");
        for (int l = 0; l < m; l++) printf("%d", Q[l]);
        printf("\n");
        for (int i = 0; i < r; i++) {
            printf("    a%-2d: ", i + 1);
            for (int j = 0; j < r; j++) printf("%d", i == j);
            printf(" ");
            for (int l = 0; l < m; l++) printf("%d", P[i][l]);
            printf("\n");
        }
        fflush(stdout);
    }
    if (stopfirst) done_flag = 1;
}

/* ------------------------------------------------------------------ */
/* per-node data for choosing row i */
typedef struct {
    int i;
    int nlin; int L[MAXL][MAXM];          /* linear: sum L_l x_l = 0 */
    int nsq;  int S2[MAXR + 1][MAXM];     /* quadratic: sum S2_l x_l^2 = 0 */
    int brk[MAXM];                         /* column l starts a new block */
    int zpre[MAXM];                        /* prefix (q, p_1..p_{i-1}) of column l is zero */
    int lexprev;                           /* need x >= P[i-1] lexicographically */
} Node;

static void rec(int i);

static int accept_full(Node *nd, int *x)
{
    int i = nd->i;
    /* ordering */
    for (int l = 1; l < m; l++) if (!nd->brk[l] && x[l] < x[l - 1]) return 0;
    for (int l = 0; l < m; l++) if (nd->zpre[l] && x[l] && !lead_ok[x[l]]) return 0;
    if (p == 7) { for (int l = 0; l < m; l++) if (x[l]) { if (!lead_ok[x[l]]) return 0; break; } }
    if (nd->lexprev) {
        for (int l = 0; l < m; l++) { if (x[l] != P[i - 1][l]) { if (x[l] < P[i - 1][l]) return 0; break; } }
    }
    /* (a) */
    int s = C[i];
    for (int l = 0; l < m; l++) s += W[l] * cube[x[l]];
    if (s % p) return 0;
    for (int t = 0; t < nd->nlin; t++) {
        int z = 0; for (int l = 0; l < m; l++) z += nd->L[t][l] * x[l];
        if (z % p) return 0;
    }
    for (int t = 0; t < nd->nsq; t++) {
        int z = 0; for (int l = 0; l < m; l++) z += nd->S2[t][l] * sq[x[l]];
        if (z % p) return 0;
    }
    return 1;
}

static void take(Node *nd, int *x)
{
    for (int l = 0; l < m; l++) P[nd->i][l] = x[l];
    rec(nd->i + 1);
}

/* Method A: DFS over columns with block monotonicity */
static void genA(Node *nd, int l, int *x)
{
    if (done_flag) return;
    if (l == m) { if (accept_full(nd, x)) take(nd, x); return; }
    int lo = (l > 0 && !nd->brk[l]) ? x[l - 1] : 0;
    for (int v = lo; v < p; v++) {
        if (nd->zpre[l] && v && !lead_ok[v]) continue;
        x[l] = v; genA(nd, l + 1, x);
    }
}

static int R[MAXL][MAXM], piv[MAXL], nr, isfree[MAXM], freel[MAXM], nfree;

static double binom(int a, int b) { double s = 1; for (int t = 1; t <= b; t++) s = s * (a - b + t) / t; return s; }

static void rec(int i)
{
    if (done_flag) return;
    nodes++;
    if (i == r) { if (final_check()) report(); return; }
    Node *nd = malloc(sizeof(Node));
    nd->i = i;
    /* blocks */
    for (int l = 0; l < m; l++) {
        int same = (l > 0) && W[l] == W[l - 1] && Q[l] == Q[l - 1];
        for (int j = 0; j < i && same; j++) if (P[j][l] != P[j][l - 1]) same = 0;
        nd->brk[l] = !same;
        int z = (Q[l] == 0);
        for (int j = 0; j < i && z; j++) if (P[j][l]) z = 0;
        nd->zpre[l] = z;
    }
    nd->lexprev = (i > 0 && C[i] == C[i - 1]);
    /* constraints on the new row x = p_i */
    nd->nlin = 0; nd->nsq = 0;
    for (int l = 0; l < m; l++) nd->L[nd->nlin][l] = md(W[l] * sq[Q[l]]);          /* (e) */
    nd->nlin++;
    for (int l = 0; l < m; l++) nd->S2[nd->nsq][l] = md(W[l] * Q[l]);              /* (f), i = j */
    nd->nsq++;
    for (int j = 0; j < i; j++) {
        for (int l = 0; l < m; l++) nd->L[nd->nlin][l] = md(W[l] * sq[P[j][l]]);   /* (b) sum w p_j^2 x */
        nd->nlin++;
        for (int l = 0; l < m; l++) nd->S2[nd->nsq][l] = md(W[l] * P[j][l]);       /* (b) sum w p_j x^2 */
        nd->nsq++;
        for (int l = 0; l < m; l++) nd->L[nd->nlin][l] = md(W[l] * Q[l] * P[j][l]); /* (f), j < i */
        nd->nlin++;
        for (int k = j + 1; k < i; k++) {
            for (int l = 0; l < m; l++) nd->L[nd->nlin][l] = md(W[l] * P[j][l] * P[k][l]); /* (c) */
            nd->nlin++;
        }
    }
    /* estimate method A size */
    double cntA = 1;
    for (int l = 0; l < m;) {
        int b = 1; while (l + b < m && !nd->brk[l + b]) b++;
        int a = nd->zpre[l] ? (p == 7 ? 3 : p) : p;
        cntA *= binom(b + a - 1, a - 1);
        l += b;
    }
    /* RREF of linear system, pivots chosen from the right */
    nr = 0;
    {
        static int M2[MAXL][MAXM];
        int rows = nd->nlin;
        for (int t = 0; t < rows; t++) memcpy(M2[t], nd->L[t], sizeof(int) * m);
        for (int l = 0; l < m; l++) isfree[l] = 1;
        for (int c = m - 1; c >= 0 && nr < rows; c--) {
            int pr = -1; for (int t = nr; t < rows; t++) if (M2[t][c]) { pr = t; break; }
            if (pr < 0) continue;
            int tmp[MAXM]; memcpy(tmp, M2[pr], sizeof(int) * m); memcpy(M2[pr], M2[nr], sizeof(int) * m); memcpy(M2[nr], tmp, sizeof(int) * m);
            int s = inv[M2[nr][c]];
            for (int l = 0; l < m; l++) M2[nr][l] = (M2[nr][l] * s) % p;
            for (int t = 0; t < rows; t++) if (t != nr && M2[t][c]) {
                int f = M2[t][c];
                for (int l = 0; l < m; l++) M2[t][l] = md(M2[t][l] - f * M2[nr][l]);
            }
            piv[nr] = c; isfree[c] = 0; nr++;
        }
        for (int t = 0; t < nr; t++) memcpy(R[t], M2[t], sizeof(int) * m);
    }
    nfree = 0; for (int l = 0; l < m; l++) if (isfree[l]) freel[nfree++] = l;
    double cntB = 1; for (int t = 0; t < nfree; t++) cntB *= p;
    int x[MAXM];
    if (cntA <= cntB) genA(nd, 0, x);
    else {
        /* genB uses globals R/piv/free which deeper levels overwrite: save them */
        int sR[MAXL][MAXM], spiv[MAXL], snr = nr, sfree[MAXM], snf = nfree, sis[MAXM];
        memcpy(sR, R, sizeof(int) * MAXM * snr); memcpy(spiv, piv, sizeof(int) * snr);
        memcpy(sfree, freel, sizeof(int) * snf); memcpy(sis, isfree, sizeof(int) * m);
        /* re-entrant wrapper */
        int vals[MAXM]; int k = 0; for (int t = 0; t < snf; t++) vals[t] = -1;
        while (snf > 0 && k >= 0 && !done_flag) {
            int l = sfree[k];
            int lo = 0;
            if (l > 0 && !nd->brk[l] && sis[l - 1]) lo = x[l - 1];
            int v = (vals[k] < 0) ? lo : vals[k] + 1;
            while (v < p && nd->zpre[l] && v && !lead_ok[v]) v++;
            if (v >= p) { vals[k] = -1; k--; continue; }
            vals[k] = v; x[l] = v;
            if (k == snf - 1) {
                for (int t = 0; t < snr; t++) {
                    int z = 0; for (int f = 0; f < snf; f++) z += sR[t][sfree[f]] * x[sfree[f]];
                    x[spiv[t]] = md(-z);
                }
                if (accept_full(nd, x)) take(nd, x);
            } else k++;
        }
        if (snf == 0) {
            for (int t = 0; t < snr; t++) x[spiv[t]] = 0;
            if (accept_full(nd, x)) take(nd, x);
        }
    }
    free(nd);
}

/* ------------------------------------------------------------------ */
/* enumerate q: nondecreasing within W-blocks, lead constraint, lambda = 1 */
static void genq(int l)
{
    if (done_flag) return;
    if (l == m) {
        int s = 0; for (int t = 0; t < m; t++) s += W[t] * cube[Q[t]];
        if (s % p != 1) return;
        rec(0);
        return;
    }
    int lo = (l > 0 && W[l] == W[l - 1]) ? Q[l - 1] : 0;
    for (int v = lo; v < p; v++) { if (v && !lead_ok[v]) continue; Q[l] = v; genq(l + 1); }
}

static int nclass;   /* number of weight classes */
static int target_n;
static void genC(int i, int cost);
static void genW(int l, int cost)
{
    if (done_flag) return;
    if (l == m) { genC(0, cost); return; }
    int lo = l ? W[l - 1] : 1;
    for (int v = lo; v <= nclass; v++) {
        int c = unitmode ? v : 1;
        if (cost + c > target_n) break;
        W[l] = v; genW(l + 1, cost + c);
    }
}
static void genC(int i, int cost)
{
    if (done_flag) return;
    if (i == r) { if (cost == target_n || !unitmode) genq(0); return; }
    int lo = i ? C[i - 1] : 1;
    for (int v = lo; v <= nclass; v++) {
        int c = unitmode ? v : 1;
        if (cost + c > target_n) break;
        C[i] = v; genC(i + 1, cost + c);
    }
}

int main(int argc, char **argv)
{
    if (argc < 3) { fprintf(stderr, "usage: %s p n [-u] [-d D] [-f] [-v]\n", argv[0]); return 1; }
    p = atoi(argv[1]); target_n = atoi(argv[2]);
    for (int a = 3; a < argc; a++) {
        if (!strcmp(argv[a], "-u")) unitmode = 1;
        else if (!strcmp(argv[a], "-f")) stopfirst = 1;
        else if (!strcmp(argv[a], "-v")) verbose = 1;
        else if (!strcmp(argv[a], "-d")) dtarget = atoi(argv[++a]);
        else if (!strcmp(argv[a], "-R")) relax = 1;    /* testing: skip (g) */
        else if (!strcmp(argv[a], "-rm")) { fix_r = atoi(argv[++a]); fix_m = atoi(argv[++a]); }
    }
    if (p != 3 && p != 5 && p != 7) { fprintf(stderr, "p must be 3, 5 or 7\n"); return 1; }
    for (int x = 0; x < p; x++) { cube[x] = x * x * x % p; sq[x] = x * x % p; lead_ok[x] = (x != 0); }
    for (int x = 1; x < p; x++) for (int y = 1; y < p; y++) if (x * y % p == 1) inv[x] = y;
    if (p == 7) { for (int x = 0; x < 7; x++) lead_ok[x] = (x == 1 || x == 3); nclass = 3; }
    else nclass = 1;
    if (p != 7) unitmode = 0;
    printf("p=%d n=%d mode=%s d>=%d\n", p, target_n, p == 7 ? (unitmode ? "unit-cost" : "weighted") : "unit", dtarget);
    clock_t t0 = clock();
    for (r = 1; r < MAXR; r++) {
        /* in unit mode n_cost >= r + m; enumerate all m with r+m <= n */
        for (m = r + 1; m < MAXM; m++) {
            if (!unitmode && r + m != target_n) continue;
            if (fix_r >= 0 && (r != fix_r || m != fix_m)) continue;
            if (r + m > target_n) break;
            long long maxpts = 1; for (int t = 0; t < r; t++) maxpts *= p;
            maxpts = (maxpts - 1) / (p - 1) - r;
            if (m > maxpts && !relax) continue;
            nodes = 0; found = 0; done_flag = 0;
            clock_t t1 = clock();
            genW(0, 0);
            printf("  r=%d m=%d: nodes=%lld codes=%lld time=%.1fs\n", r, m, nodes, found,
                   (double)(clock() - t1) / CLOCKS_PER_SEC);
            fflush(stdout);
            if (stopfirst && found) goto end;
        }
    }
end:
    printf("TOTAL p=%d n=%d codes=%lld time=%.1fs\n", p, target_n, found_total, (double)(clock() - t0) / CLOCKS_PER_SEC);
    return 0;
}
