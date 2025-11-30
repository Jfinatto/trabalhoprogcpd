#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h> 

// Macro para acessar o elemento (i, j) em uma matriz 1D
#define IDX(i, j, width) ((i) * (width) + (j))

/**
 * @brief Preenche uma matriz com valores aleatórios (entre 0 e 1).
 */
void preencher_matriz(double *mat, int size) {
    for (int i = 0; i < size; i++) {
        mat[i] = (double)rand() / (double)RAND_MAX;
    }
}

/**
 * @brief Calcula a multiplicação para uma fatia da matriz A.
 * Usa a ordem de loop (i, k, j) para melhor performance de cache.
 * * @param A_chunk    Ponteiro para a fatia local de A (chunk_rows x N)
 * @param B_full     Ponteiro para a matriz B completa (N x N)
 * @param C_chunk    Ponteiro para a fatia local de C (resultado) (chunk_rows x N)
 * @param chunk_rows Número de linhas na fatia local
 * @param N          Dimensão total da matriz (N x N)
 */
void multiplicar_chunk(double *A_chunk, double *B_full, double *C_chunk, int chunk_rows, int N) {
    // Ordem de loop i-k-j para otimização de cache
    // A_chunk[i][k] é acessado sequencialmente
    // B_full[k][j] é acessado sequencialmente (dentro do loop j)
    for (int i = 0; i < chunk_rows; i++) {
        for (int k = 0; k < N; k++) {
            // Guarda A_chunk[i][k] em um registrador (provavelmente)
            double A_ik = A_chunk[IDX(i, k, N)]; 
            
            for (int j = 0; j < N; j++) {
                // Na primeira iteração de k, inicializa C
                if (k == 0) {
                    C_chunk[IDX(i, j, N)] = 0.0;
                }
                C_chunk[IDX(i, j, N)] += A_ik * B_full[IDX(k, j, N)];
            }
        }
    }
}


int main(int argc, char *argv[]) {
    
    int rank, num_procs; // Rank do processo e número total de processos
    int N;               // Dimensão da matriz (N x N)
    int rows_per_proc;   // Quantas linhas cada processo calculará
    
    double *A_full = NULL; // Matriz A completa (só no mestre)
    double *B_full = NULL; // Matriz B completa (alocada por todos)
    double *C_full = NULL; // Matriz C completa (só no mestre)
    
    double *A_chunk;       // Fatia local de A
    double *C_chunk;       // Fatia local de C
    
    double start_time, end_time, tempo_total;

    // --- 1. Inicialização do MPI ---
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    
    // Pega a dimensão N da linha de comando
    if (rank == 0) {
        if (argc != 2) {
            fprintf(stderr, "Uso: mpirun -np <num_procs> %s <N>\n", argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        N = atoi(argv[1]);
        
        // Validação simples
        if (N % num_procs != 0) {
            fprintf(stderr, "Erro: N (%d) deve ser divisível pelo número de processos (%d)\n", N, num_procs);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        printf("--- Multiplicação de Matrizes MPI (%dx%d) com %d processos ---\n", N, N, num_procs);
        printf("[Mestre] Etapa: Alocando matrizes A, B, C...\n");
    }

    // --- 2. Broadcast de N e Alocação de Memória ---
    
    // Mestre envia N para todos os escravos
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Mestre calcula quantas linhas por processo
    rows_per_proc = N / num_procs;

    // TODOS os processos alocam a matriz B completa (pois todos precisam dela)
    B_full = (double*)malloc(N * N * sizeof(double));
    
    // TODOS os processos alocam suas fatias de A e C
    A_chunk = (double*)malloc(rows_per_proc * N * sizeof(double));
    C_chunk = (double*)malloc(rows_per_proc * N * sizeof(double));
    
    if (B_full == NULL || A_chunk == NULL || C_chunk == NULL) {
        fprintf(stderr, "[Rank %d] Erro ao alocar memória para chunks!\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    // --- 3. Preenchimento (Só no Mestre) ---
    if (rank == 0) {
        // Mestre aloca A e C completos
        A_full = (double*)malloc(N * N * sizeof(double));
        C_full = (double*)malloc(N * N * sizeof(double));
        if (A_full == NULL || C_full == NULL) {
            fprintf(stderr, "[Mestre] Erro ao alocar memória para matrizes A/C!\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        printf("[Mestre] Etapa: Preenchendo matrizes A e B com valores aleatórios...\n");
        srand(time(NULL));
        preencher_matriz(A_full, N * N);
        preencher_matriz(B_full, N * N); // Mestre preenche a sua cópia de B
        printf("[Mestre] Etapa: Preenchimento concluído.\n");
    }
    
    // Aguarda todos os processos chegarem aqui antes de iniciar o timer
    MPI_Barrier(MPI_COMM_WORLD);

    // --- 4. Distribuição dos Dados e Início do Timer ---
    
    if (rank == 0) {
        printf("[Mestre] Etapa: Iniciando cálculo (Broadcast de B, Scatter de A)...\n");
        // Inicia o timer DEPOIS da alocação/preenchimento, ANTES da comunicação
        start_time = MPI_Wtime();
    }
    
    // Mestre (rank 0) envia a matriz B completa para todos os outros
    // A fonte é B_full no rank 0, o destino é B_full em todos os outros
    MPI_Bcast(B_full,         // buffer
              N * N,          // contagem
              MPI_DOUBLE,     // tipo
              0,              // root (mestre)
              MPI_COMM_WORLD);
              
    // Mestre (rank 0) distribui (espalha) as linhas de A_full
    // Cada processo (incluindo o mestre) recebe sua fatia em A_chunk
    MPI_Scatter(A_full,             // buffer de envio (só usado no mestre)
                rows_per_proc * N,  // contagem de envio (por processo)
                MPI_DOUBLE,         // tipo
                A_chunk,            // buffer de recebimento (usado por todos)
                rows_per_proc * N,  // contagem de recebimento
                MPI_DOUBLE,         // tipo
                0,                  // root (mestre)
                MPI_COMM_WORLD);

    // --- 5. Cálculo Paralelo ---
    
    // Agora TODOS os processos (incluindo o mestre) têm os dados necessários
    // Cada um calcula sua própria fatia
    multiplicar_chunk(A_chunk, B_full, C_chunk, rows_per_proc, N);

    // --- 6. Coleta dos Resultados (Gather) ---

    // Todos os processos enviam seus C_chunk de volta para o C_full do Mestre
    MPI_Gather(C_chunk,            // buffer de envio (usado por todos)
               rows_per_proc * N,  // contagem de envio
               MPI_DOUBLE,         // tipo
               C_full,             // buffer de recebimento (só usado no mestre)
               rows_per_proc * N,  // contagem de recebimento (por processo)
               MPI_DOUBLE,         // tipo
               0,                  // root (mestre)
               MPI_COMM_WORLD);
               
    // --- 7. Finalização e Exibição do Tempo ---

    // Para o timer assim que o Mestre recebeu o último resultado
    if (rank == 0) {
        end_time = MPI_Wtime();
        tempo_total = end_time - start_time;
        
        printf("[Mestre] Etapa: Resultados coletados.\n");
        printf("[Mestre] Etapa: Execução concluída.\n");
        printf("\n======================================================\n");
        printf("Tempo de execução do cálculo (MPI): %f segundos\n", tempo_total);
        printf("======================================================\n");
    }
    
    // --- 8. Liberação de Memória ---
    if (rank == 0) {
        printf("[Mestre] Etapa: Liberando memória...\n");
        free(A_full);
        free(C_full);
    }
    
    // Todos liberam os buffers que alocaram
    free(A_chunk);
    free(B_full);
    free(C_chunk);
    
    MPI_Finalize();
    return 0;
}
