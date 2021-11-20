#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <pmmintrin.h>

void print128_num(__m128 var)
{
    u_int16_t val[8];
    memcpy(val, &var, sizeof(val));
    printf("Numerical: %i %i %i %i %i %i %i %i \n", 
           val[0], val[1], val[2], val[3], val[4], val[5], 
           val[6], val[7]);
}


int main( int argc, char *argv[] ) {

    int m, n, test, i, j;
    float alfa;
    struct timeval t0, t1, t;

    // Parámetro 1 -> m, filas de A
    // Parámetro 2 -> n, columnas de A y numero de elementos de B
    // Parámetro 3 -> alfa
    // Parámetro 4 -> booleano que nos indica si se desea imprimir matrices y vectores de entrada y salida
    if(argc>3){
        m = atoi(argv[1]);
        n = atoi(argv[2]);
        alfa = atof(argv[3]);
        test = atoi(argv[4]);
    }
    else{
        printf("NUMERO DE PARAMETROS INCORRECTO\n");
	printf("Uso: ./matrizVectorP1 <n> <m> <alfa> <debug>\n");
	printf("\t<m>\t-> número de filas de la matriz\n");
	printf("\t<n>\t-> número de columnas de la matriz / elementos del vector\n");
	printf("\t<alfa>\t-> parámetro de escalado\n" );
	printf("\t<debug>\t-> indica si deben imprimirse las matrices y vectores de entrada y salida (0/1)\n");
        exit(0);
    }

    float *x = (float *)malloc(n*sizeof(float));
    float *A = (float *)malloc(m*n*sizeof(float));
    float *y = (float *)malloc(m*sizeof(float));

    // Se inicializan la matriz y los vectores

    for(i=0; i<m; i++){
        for(j=0; j<n; j++){
            A[i*n+j] = ((float)(1+i+j))/m/n; // A[i*n+j-1]+1; 
        }
    }

    for(i=0; i<n; i++){
        x[i] =  ((float)(1+i))/n; //i + 1; 
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

    //gcc -O3 -march=native matrizVectorP1.c

    float *vY;
    vY = (float *) _mm_malloc(sizeof(float)*n,16);
    int count = 0;

    for ( int k = 0; k < m; k++) {
        for ( int i = 0; i < n; i+=4) {
            __m128 reg_A = _mm_load_ps(&A[k*n+i]);
            __m128 reg_X = _mm_load_ps(&x[i]);

            __m128 reg_result = _mm_mul_ps(reg_A, reg_X);

            reg_result = _mm_hadd_ps(reg_result, reg_result);
            __m128 reg_Y = _mm_hadd_ps(reg_result, reg_result);

            _mm_store_ps(&vY[i], reg_Y);
            
            y[count] += vY[i];
        }
        count += 1;
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




