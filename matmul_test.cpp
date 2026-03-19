void MatmulNaive(const Matrix<double>& A, const Matrix<double>& B, const Matrix<double>& C){
    auto M = A.col();  


    auto K = A.row();  


    for(int i=0;i<M;i++){
        for(int j=0;j<N;j++){
            for(int k=0;k<K;k++){
                C(i,j) += A(i,k) * B(k,j); 

            }
        }
    }

}

void MatmulNaive_Order(const Matrix<double>& A, const Matrix<double>& B, const Matrix<double>& C){
    auto M = A.col();
    auto N = B.col();
    auto K = A.row(); 


    for (int i=0; i<M; i++){        
        for (int k=0; k<K; k++){   
            for(int j=0; j<N; j++){ 
                C(i,j) += A(i,k) * B(k,j);

            }
        }
    }
}

template<int BLOCK>
void naive_block(const double *a, const double *mb, double *c, int N, int K){
    for(int i=0;i<BLOCK;i++, c+=N, a+=K){
        const double* b=mb;
        for(int k=0;k<BLOCK;k++, b+=N){
            for(int j=0;j<BLOCK;j++){
                c[j] += a[k] * b[j];

            }
        }
    }
}

void MatmulNaive_Tile(const Matrix<double>& A, const Matrix<double>& B, const Matrix<double>& C, int tile_size){
    auto M = A.row();
    auto N = B.col();
    auto K = A.row();

    constexpr auto BLOCK=64; /* 
    """64x64 boyutunda blok kullanıyoruz.
    Cache boyutuna göre seçilmiş, L1 veya L2 cache'de rahatça sığacak boyut."""
    */

    for(int ib=0; ib<M; ib+=BLOCK){
        for(int kb=0; kb<K; kb+=BLOCK){
            for(int jb=0; jb<N; jb+=BLOCK){
                const double* a= A(ib, kb);
                const double* mb= B(kb, jb);
                double* c= C(ib, jb);

                ikernel::naive_block<BLOCK>(a, mb, c, N, K);

            }
        }
    }
}

// SSE (128-bit SIMD) ile block çarpımı
template<int BLOCK>
void simd_block(const double *a, const double *mb, double *c, int N, int K){
    constexpr int simd_doubles=128/(sizeof(double)*8); 

    for(int i=0;i<BLOCK;i++, c+=N, a+=K){
        const double* b=mb;
        for(int k=0;k<BLOCK;k++, b+=N){
            __m128d a_reg = _mm_load_sd(&a[k]);
            a_reg         = _mm_unpacklo_pd(a_reg, a_reg);
            for(int j=0;j<BLOCK;j+=simd_doubles){
                __m128d b_reg = _mm_load_pd(&b[j]);
                __m128d c_reg = _mm_load_pd(&c[j]);
                c_reg = _mm_add_pd(_mm_mul_pd(b_reg, a_reg), c_reg);
                _mm_store_pd(&c[j], c_reg);

            }
        }
    }
}

template<int BLOCK>
void avx_block(const double *a, const double *mb, double *c, int N, int K){
    constexpr int avx_doubles=256/(sizeof(double)*8); // 4 double aynı anda
    for(int i=0;i<BLOCK;i++, c+=N, a+=K){
        const double* b=mb;
        for(int k=0;k<BLOCK;k++, b+=N){
            __m256d a_reg = _mm256_broadcast_sd(&a[k]); 
            for(int j=0;j<BLOCK;j+=avx_doubles){
                __m256d b_reg = _mm256_load_pd(&b[j]);
                __m256d c_reg = _mm256_load_pd(&c[j]);
                c_reg = _mm256_add_pd(_mm256_mul_pd(b_reg, a_reg), c_reg);
                _mm256_store_pd(&c[j], c_reg);
            }
        }
    }
}

// Cache ve AVX ile optimize edilmiş matris çarpımı
void matMul_Avx_Cache(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C){
    auto M = A.row();
    auto K = A.col();
    auto N = B.col();
    constexpr auto Mc=180, Nc=96, Kc=240, Nr=12, Mr=4; 

    for(int ib=0; ib<M; ib+=Mc){
        for(int kb=0; kb<K; kb+=Kc){
            for(int jb=0; jb<N; jb+=Nc){
                const double* ma = &A(ib, kb);
                const double* mb = &B(kb, jb);
                double* mc = &C(ib, jb);
                for(int i2=0; i2<Mc; i2+=Mr){
                    for(int j2=0; j2<Nc; j2+=Nr){
                        const double* a = &ma[i2*K];
                        const double* b = &mb[j2];
                        double* c = &mc[i2*N+j2];
                        ikernels::avx_block<Nr, Mr, Kc, Nc>(c, a, b, N, K);

                    }
                }
            }
        }
    }
}

int main(){
    int M =2, N=4, K=2;
    Matrix<double> A(M, K);
    Matrix<double> B(K, N);
    Matrix<double> C(M, N);

    for(int i=0;i<M;i++){
        for(int j=0;j<K;j++){
            A(i,j) = i+j+1; 
        }
    }
    for(int j=0;j<K;j++){
        for(int k=0;k<N;k++){
            B(j,k) = j+k+1; 
        }
    }
    MatmulNaive(A, B, C);

    for(int i=0;i<M;i++){
        for(int j=0;j<N;j++){
            std::cout << C(i,j)<< " ";  
        }
        std::cout << std::endl;
    }

}