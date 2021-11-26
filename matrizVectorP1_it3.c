#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <pmmintrin.h>
#include <immintrin.h>
#include <unistd.h>

#define _MM_TRANSPOSE8_PS(row0, row1, row2, row3, row4, row5, row6, row7) \
	do { \
		__m256 __t0, __t1, __t2, __t3, __t4, __t5, __t6, __t7; \
		__m256 __tt0, __tt1, __tt2, __tt3, __tt4, __tt5, __tt6, __tt7; \
		__t0 = _mm256_unpacklo_ps(row0, row1); \
		__t1 = _mm256_unpackhi_ps(row0, row1); \
		__t2 = _mm256_unpacklo_ps(row2, row3); \
		__t3 = _mm256_unpackhi_ps(row2, row3); \
		__t4 = _mm256_unpacklo_ps(row4, row5); \
		__t5 = _mm256_unpackhi_ps(row4, row5); \
		__t6 = _mm256_unpacklo_ps(row6, row7); \
		__t7 = _mm256_unpackhi_ps(row6, row7); \
		__tt0 = _mm256_shuffle_ps(__t0, __t2, _MM_SHUFFLE(1, 0, 1, 0)); \
		__tt1 = _mm256_shuffle_ps(__t0, __t2, _MM_SHUFFLE(3, 2, 3, 2)); \
		__tt2 = _mm256_shuffle_ps(__t1, __t3, _MM_SHUFFLE(1, 0, 1, 0)); \
		__tt3 = _mm256_shuffle_ps(__t1, __t3, _MM_SHUFFLE(3, 2, 3, 2)); \
		__tt4 = _mm256_shuffle_ps(__t4, __t6, _MM_SHUFFLE(1, 0, 1, 0)); \
		__tt5 = _mm256_shuffle_ps(__t4, __t6, _MM_SHUFFLE(3, 2, 3, 2)); \
		__tt6 = _mm256_shuffle_ps(__t5, __t7, _MM_SHUFFLE(1, 0, 1, 0)); \
		__tt7 = _mm256_shuffle_ps(__t5, __t7, _MM_SHUFFLE(3, 2, 3, 2)); \
		row0 = _mm256_permute2f128_ps(__tt0, __tt4, 0x20); \
		row1 = _mm256_permute2f128_ps(__tt1, __tt5, 0x20); \
		row2 = _mm256_permute2f128_ps(__tt2, __tt6, 0x20); \
		row3 = _mm256_permute2f128_ps(__tt3, __tt7, 0x20); \
		row4 = _mm256_permute2f128_ps(__tt0, __tt4, 0x31); \
		row5 = _mm256_permute2f128_ps(__tt1, __tt5, 0x31); \
		row6 = _mm256_permute2f128_ps(__tt2, __tt6, 0x31); \
		row7 = _mm256_permute2f128_ps(__tt3, __tt7, 0x31); \
	} while (0)


__m256 cosa(float *A, float *x, int row, int n, int column, int alfa, int sumI, int sumK) {
    __m256 reg_A;
    __m256 reg_X;
    if (sumI < 8) { //si las columnas del vector son menores a 8 se rellenan las faltantes con ceros hasta llegar a 8 columnas
        float *Ac = (float *) _mm_malloc(sizeof(float)*8,32);
        for (int j = 0; j < 8; j++) {
            if (j >= sumI) {
                Ac[j] = 0;
            } else {
                Ac[j] = A[row*n+column+j];
            }
        }
        reg_A = _mm256_load_ps(&Ac[0]);
        reg_X = _mm256_load_ps(&x[column]);
        _mm_free(Ac);
    } else { //si el array tiene 8 elementos proseguimos de forama normal
        reg_A = _mm256_loadu_ps(&A[row*n+column]);
        reg_X = _mm256_load_ps(&x[column]);
    }

    __m256 reg_result = _mm256_mul_ps(reg_A, reg_X); //multiplico el array X por el vector de 8 elementos procedente de la fila de la matriz

    const __m256 scalar = _mm256_set1_ps(alfa); 
    __m256 result = _mm256_mul_ps(reg_result, scalar); //multiplico el resultado por alfa

    return result;
}

int main( int argc, char *argv[] ) {

    int m, n, test, i, j;
    float alfa;
    struct timeval t0, t1, t;

    // Parámetro 1 -> m, filas de A
    // Parámetro 2 -> n, columnas de A y numero de elementos de B
    // Parámetro 3 -> alfa
    // Parámetro 4 -> booleano que nos indica si se desea imprimir matrices y vectores de entrada y salida
    if(argc>4){
        m = atoi(argv[1]);
        n = atoi(argv[2]);
        alfa = atof(argv[3]);
        test = atoi(argv[4]);
    } else {
        printf("NUMERO DE PARAMETROS INCORRECTO\n");
        printf("Uso: ./matrizVectorP1 <n> <m> <alfa> <debug>\n");
        printf("\t<m>\t-> número de filas de la matriz\n");
        printf("\t<n>\t-> número de columnas de la matriz / elementos del vector\n");
        printf("\t<alfa>\t-> parámetro de escalado\n" );
        printf("\t<debug>\t-> indica si deben imprimirse las matrices y vectores de entrada y salida (0/1)\n");
        exit(0);
    }

    float *x = (float *) _mm_malloc(sizeof(float)*n,32);
    float *A = (float *) _mm_malloc(sizeof(float)*m*n,32);
    float *y = (float *) _mm_malloc(sizeof(float)*m,32);


    // Se inicializan la matriz y los vectores

    for(i=0; i<m; i++) {
        for(j=0; j<n; j++) {
            A[i*n+j] = ((float)(1+i+j))/m/n; // 
        }
    }

    for(i=0; i<n; i++){
        x[i] =  ((float)(1+i))/n; //
    }

    for(i=0; i<m; i++){
        y[i] = ((float)(1-i))/m;
    }
    
    if(test){
        printf("\nMatriz A es...\n");
        for(i=0; i<m; i++){
            for(j=0; j<n; j++){
                printf("%.2f ", A[i*n+j]);
            }
            printf("\n");
        }

        printf("\nVector x es...\n");
        for(i=0; i<n; i++){
            printf("%.2f ", x[i]);
        }
        printf("\n");

        printf("\nVector y al principio es...\n");
        for(i=0; i<m; i++){
            printf("%.2f ", y[i]);
        }
        printf("\n");
    }

    // Parte fundamental del programa
    assert (gettimeofday (&t0, NULL) == 0);
    
    //gcc -march=native matrizVectorP1.c -g

    float *vY;
    __m256 *rowArray;
    vY = (float *) _mm_malloc(sizeof(float)*n,32);
    rowArray = (__m256 *) _mm_malloc(sizeof(__m256)*8,32);
    
    int count = 0;
    int sumK = 8;
    int sumI = 8;

    for ( int k = 0; k < m; k+=sumK) { //itera sobre las filas de la matriz, de sumK en sumK
        sumK = 8;
        if (m-k < 8) //si las filas restantes son menores a 8, sumK se ajusta al numero restante
            sumK = m-k;
        for ( int i = 0; i < n; i+=sumI) { //itera sobre las columnas de la matriz, de sumI en sumI
            sumI = 8;
            if (n-i < 8)  //si las columnas restantes son menores a 8, sumI se ajusta al numero restante
                sumI = n-i;
                
            for (int j = 0; j < 8; j++) { //ahora que estamos en un bloque sumK*sumI, podemos iterar sobre las 8 filas del bloque
                __m256 row;
                if (j < sumK) { //si el iterador es menor que el número de filas del bloque, todo correcto
                    row = cosa(A, x, k+j, n, i, alfa, sumI, sumK);
                } else {
                    row = _mm256_set1_ps(0);  //si el iterador es mayor al numero de filas del bloque, rellenamos la fila con ceros
                }
                rowArray[j] = row; //array de 8 elementos que contiene 8 arrays __mm256
            }

            _MM_TRANSPOSE8_PS(rowArray[0], rowArray[1], rowArray[2], rowArray[3], rowArray[4], rowArray[5], rowArray[6], rowArray[7]); //transponemos la matriz 8x8 formada por los arrays

            __m256 res1 = _mm256_add_ps(rowArray[0], rowArray[1]); //sumo de forma vertical las columnas de la matriz resultante de la transposición
            __m256 res2 = _mm256_add_ps(rowArray[2], rowArray[3]);
            __m256 res3 = _mm256_add_ps(rowArray[4], rowArray[5]);
            __m256 res4 = _mm256_add_ps(rowArray[6], rowArray[7]);

            __m256 result1 = _mm256_add_ps(res1, res2);
            __m256 result2 = _mm256_add_ps(res3, res4);
            __m256 result = _mm256_add_ps(result1, result2); 

            _mm256_store_ps(&vY[0], result);
            
            for (int j = 0; j < 8; j++) {
                y[count+j] += vY[j]; //itero sobre el resultado para sumarlo al total
            }  
        }
        count += sumK; //sumo sumK a count

    }


    assert (gettimeofday (&t1, NULL) == 0);
    timersub(&t1, &t0, &t);

    if(test){
        printf("\nAl final vector y es...\n");
        for(i=0; i<m; i++){
            printf("%.2f ", y[i]);
        }
        printf("\n");

        float *testy = (float *) malloc(m*sizeof(float));
        for(i=0; i<m; i++){
            testy[i] = ((float)(1-i))/m;
        }

        // Se calcula el producto sin ninguna vectorización
        for (i=0; i<m; i++) {
            for (j=0; j<n; j++) {
                testy[i] += alfa*A[i*n+j]*x[j];
            }
        }

        int errores = 0;
        for(i=0; i<m; i++){
	    if( (testy[i]-y[i])*(testy[i]-y[i]) > 1e-10 ) {
                errores++;
                printf("\n Error en la posicion %d porque %f != %f", i, y[i], testy[i]);
            }
        }
        printf("\n%d errores en el producto matriz vector con dimensiones %dx%d\n", errores, m, n);
        free(testy);
    }

    printf ("Tiempo      = %ld:%ld(seg:mseg)\n", t.tv_sec, t.tv_usec/1000);

    free(x);
    free(y);
    free(A);
	
    return 0;
}




