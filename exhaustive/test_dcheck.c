/* Validation of the -d check of triorth.c on known codes.
   Reads known_codes.txt (systematic form A=(I_r|P), h=(0|q)) and, for each code of
   distance d, checks that final_check() accepts with dtarget = 3..d and rejects with d+1.
   Compile: gcc -O2 -o test_dcheck test_dcheck.c   (triorth.c in the same directory) */
#define main triorth_main
#include "triorth.c"
#undef main
int main(void) {
    FILE *f = fopen("known_codes.txt", "r"); if (!f) { perror("known_codes.txt"); return 1; }
    int pp, rr, mm, dd; char name[64]; int fails = 0;
    while (fscanf(f, "%d %d %d %d %63s", &pp, &rr, &mm, &dd, name) == 5) {
        p = pp; r = rr; m = mm;
        for (int x = 1; x < p; x++) for (int y = 1; y < p; y++) if (x * y % p == 1) inv[x] = y;
        for (int l = 0; l < m; l++) fscanf(f, "%d", &Q[l]);
        for (int i = 0; i < r; i++) for (int l = 0; l < m; l++) fscanf(f, "%d", &P[i][l]);
        printf("%-6s p=%d n=%2d d=%d :", name, p, r + m, dd);
        for (int t = 3; t <= dd + 1; t++) {
            dtarget = t; int acc = final_check(); int ok = (t <= dd) ? acc : !acc;
            printf("  -d %d %s%s", t, acc ? "accept" : "reject", ok ? "" : "(WRONG)");
            if (!ok) fails++;
        }
        printf("\n");
    }
    printf(fails ? "FAILURES: %d\n" : "all checks passed\n", fails);
    return fails != 0;
}
