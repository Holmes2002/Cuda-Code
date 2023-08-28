#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#include "cpu.cu"
#include "cuda.cu"

#define M 600
#define N 600

void initialize_grid(double w[M][N]);

int main(int argc, char *argv[])

{
    double epsilon;
    int i;
    int j;
    FILE *output;
    char output_filename[80];
    int success;
    double w[M][N];

    printf("\n");
    printf("HEATED_PLATE <epsilon> <fichero-salida>\n");
    printf("  C/serie version\n");
    printf("  A program to solve for the steady state temperature distribution\n");
    printf("  over a rectangular plate.\n");
    printf("\n");
    printf("  Spatial grid of %d by %d points.\n", M, N);

    //
    //  Read EPSILON from the command line or the user.
    //
    epsilon = atof(argv[1]);
    printf("The iteration will be repeated until the change is <= %lf\n", epsilon);

    //
    //  Read OUTPUT_FILE from the command line or the user.
    //
    success = sscanf(argv[2], "%s", output_filename);
    if (success != 1)
    {
        printf("\n");
        printf("HEATED_PLATE\n");
        printf(" Error reading output file name\n");
        return 1;
    }

    printf("  The steady state solution will be written to %s\n", output_filename);

    initialize_grid(w);
    calculate_solution_para(w, epsilon);

    initialize_grid(w);
    calculate_solution_seq(w, epsilon);


    //  Write the solution to the output file.
    output = fopen(output_filename, "wt");

    fprintf(output, "%d\n", M);
    fprintf(output, "%d\n", N);

    for (i = 0; i < M; i++)
    {
        for (j = 0; j < N; j++)
        {
            fprintf(output, "%lg ", w[i][j]);
        }
        fprintf(output, "\n");
    }
    fclose(output);

    printf("\n");
    printf(" Solution written to file %s\n", output_filename);
    //
    //  Terminate.
    //
    printf("\n");
    printf("HEATED_PLATE_Serie:\n");
    printf("  Normal end of execution.\n");

    return 0;
}

void initialize_grid(double w[M][N])
{
    int i;
    int j;

    //
    //  Set the boundary values, which don't change.
    //
    for (i = 1; i < M - 1; i++)
        w[i][0] = 0.0;

    for (i = 1; i < M - 1; i++)
        w[i][N - 1] = 0.0;

    for (j = 0; j < N; j++)
        w[M - 1][j] = 0.0;

    for (j = 0; j < N; j++)
        w[0][j] = 0.0;

    //  Initialize the interior solution to the mean value.

    for (i = 1; i < M - 1; i++)
        for (j = 1; j < N - 1; j++)
            w[i][j] = 0;

    // Initialize Heat Area.

    for (i = 0; i < 50; i++)
        for (j = 0; j < 50; j++)
            w[i][j] = 100.0;
}
