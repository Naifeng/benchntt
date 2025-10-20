#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "config.h"

void load_twiddles(uint64_t* twd_s, int N){
    if(N <= 1024) {
        char x_file_name[100];
        char twd_file_name[100];
        sprintf(twd_file_name, "../data/twd_%d.txt", N);
        FILE *twd_file = fopen(twd_file_name, "r");
        if(twd_file == NULL){
            printf("file open error\n");
            exit(1);
        }

        uint64_t hi, lo;
        for(int i = 0; i < N; i++){
            if(fscanf(twd_file, "%lu", &hi) != 1){
                printf("Error reading %d twd file\n", N);
                fclose(twd_file);
                exit(1);
            }
            if(fscanf(twd_file, "%lu", &lo) != 1){
                printf("Error reading %d twd file\n", N);
                fclose(twd_file);
                exit(1);
            }
            twd_s[2*i] = hi;
            twd_s[2*i+1] = lo;
        }
        fclose(twd_file);
    } else {
        uint64_t* twd = (uint64_t*)(malloc(sizeof(uint64_t)*(1024*2)));

        char twd_file_name[100];
        sprintf(twd_file_name, "../data/twd_%d.txt", 1024);

        FILE *twd_file = fopen(twd_file_name, "r");
        if(twd_file == NULL){
            printf("file open error\n");
            exit(1);
        }

        uint64_t hi, lo;
        for(int i = 0; i < 1024; i++){
            if(fscanf(twd_file, "%lu", &hi) != 1){
                printf("Error reading %d twd file\n", N);
                fclose(twd_file);
                free(twd);
                exit(1);
            }
            if(fscanf(twd_file, "%lu", &lo) != 1){
                printf("Error reading %d twd file\n", N);
                fclose(twd_file);
                free(twd);
                exit(1);
            }
            twd[2*i] = hi;
            twd[2*i+1] = lo;
        }
        fclose(twd_file);
    
        // duplicate input data
        for(int i = 0; i < 1024; i++){
            for(int j = 0; j < N; j += 1024){
                twd_s[2*(i+j)] = twd[2*(i)];
                twd_s[2*(i+j)+1] = twd[2*(i)+1];
            }
        }
        free(twd);
    }
}

void load_test_inputs(uint64_t* x, int N){
    if(N <= 1024) {
        char x_file_name[100];
        sprintf(x_file_name, "../data/x_%d.txt", N);
        FILE *x_file = fopen(x_file_name, "r");
        if(x_file == NULL){
            printf("file open error\n");
            exit(1);
        }

        uint64_t hi, lo;
        for(int i = 0; i < N; i++){
            if(fscanf(x_file, "%lu", &hi) != 1){
                printf("Error reading x file\n");
                fclose(x_file);
                exit(1);
            }
            if(fscanf(x_file, "%lu", &lo) != 1){
                printf("Error reading x file\n");
                fclose(x_file);
                exit(1);
            }
            x[2*i] = hi;
            x[2*i+1] = lo;
        }
        fclose(x_file);
    } else {
        uint64_t* x_tmp = (uint64_t*)(malloc(sizeof(uint64_t)*(1024*2)));

        char x_file_name[100];
        sprintf(x_file_name, "../data/x_%d.txt", 1024);

        FILE *x_file = fopen(x_file_name, "r");
        if(x_file == NULL){
            printf("file open error\n");
            exit(1);
        }

        uint64_t hi, lo;
        for(int i = 0; i < 1024; i++){
            if(fscanf(x_file, "%lu", &hi) != 1){
                printf("Error reading x file\n");
                fclose(x_file);
                free(x_tmp);
                exit(1);
            }
            if(fscanf(x_file, "%lu", &lo) != 1){
                printf("Error reading x file\n");
                fclose(x_file);
                free(x_tmp);
                exit(1);
            }
            x_tmp[2*i] = hi;
            x_tmp[2*i+1] = lo;
        }
        fclose(x_file);
    
        // duplicate input data
        for(int i = 0; i < 1024; i++){
            for(int j = 0; j < N; j+=1024){
                x[2*(i+j)] = x_tmp[2*(i)];
                x[2*(i+j)+1] = x_tmp[2*(i)+1];
            }
        }
        free(x_tmp);
    }
}

void load_test_outputs(uint64_t* y, int N){
    if(N <= 1024) {
        char y_file_name[100];
        sprintf(y_file_name, "../data/ver_%d.txt", N);
        FILE *y_file = fopen(y_file_name, "r");
        if(y_file == NULL){
            printf("file open error\n");
            exit(1);
        }

        for(int i = 0; i < N; i++){
            uint64_t hi, lo;
            if(fscanf(y_file, "%lu", &hi) != 1){
                printf("Error reading y file\n");
                fclose(y_file);
                exit(1);
            }
            if(fscanf(y_file, "%lu", &lo) != 1){
                printf("Error reading y file\n");
                fclose(y_file);
                exit(1);
            }
            y[2*i] = hi;
            y[2*i+1] = lo;
        }
        fclose(y_file);
    } else {
        printf("There is no test output data for N = %d\n", N);
        exit(0);
    }
}

void load_test_blas(uint64_t* y, char* file_name){
    FILE *y_file = fopen(file_name, "r");
    if(y_file == NULL){
        printf("file open error\n");
        exit(1);
    }

    for(int i = 0; i < 1024; i++){
        uint64_t hi, lo;
        if(fscanf(y_file, "%lu", &hi) != 1){
            printf("Error reading y file\n");
            fclose(y_file);
            exit(1);
        }
        if(fscanf(y_file, "%lu", &lo) != 1){
            printf("Error reading y file\n");
            fclose(y_file);
            exit(1);
        }
        y[2*i] = hi;
        y[2*i+1] = lo;
    }
    fclose(y_file);
}