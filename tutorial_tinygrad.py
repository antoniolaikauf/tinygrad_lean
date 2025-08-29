from tinygrad import Tensor

a = Tensor.empty(4, 4)
b = Tensor.empty(4, 4)
print((a+b).tolist())

# comand python3 tutorial_tinygrad.py

# *** CPU        1 E_4_4                                        arg  3 mem  0.00 GB tm     30.04us/     0.03ms (     0.00 GFLOPS    0.0|0.0     GB/s) ['tolist', '__add__', 'empty']
# [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]

# E sta per elementwise
# _4_4 shape della computazione
# arg   3 sarebbero gli argomenti a, b e l'output quindi 3 
# 0.00 GB quanta memoria è stata usata, questa è zero essendo che viene azzerata a 0 quando si utilizzano tensori di piccole dimensioni
# 30.04us/     0.03ms il primo valore dice quanto tempo ci è voluto per fare la computazione, invece il secondo parametro dice quanto tempo è trascorso dall'esecuzione del programma
# 0.00 GFLOPS quante operazioni con virgola mobile sono state eseguite 


# comand DEBUG=4 python3 tutorial_tinygrad.py 

'''
restrict non permette ad altri puntatori di non accedere alla area di memoria se non un puntatore associato a quella variabile 

int alu0 = (ridx0<<2); usato come ofset, viene usato come indice per ottenere un pezzo di memoria degli array

float4 val0 = (*((float4*)((data1_16+alu0))));
float4 val1 = (*((float4*)((data2_16+alu0))));
val0 e val1 sono blocchi di memoria di 4 valori, questo grazie alla combinazione tra alu0 e float4.
il primo permette di spostare il puntatore di 4 byte ogni volta essendo che alu0 avrà come valore 0, 4, 8, 12
le matrici sono vettori di 16 byte quindi per esempio nel primo ciclo parte dal puntatore 0 perchè alu0 è 0 e 
dovrebbe prendere tutti i valori dell'array ma grazie ad byte4 ne prende solo quattro.
alla fine si crea un tensore data0_16 che consiste nel 3 elemento come detto da arg3 sopra 


typedef float float4 __attribute__((aligned(16),vector_size(16)));
void E_4_4(float* restrict data0_16, float* restrict data1_16, float* restrict data2_16) {
  for (int ridx0 = 0; ridx0 < 4; ridx0++) {
    
    int alu0 = (ridx0<<2);
    float4 val0 = (*((float4*)((data1_16+alu0))));
    float4 val1 = (*((float4*)((data2_16+alu0))));
    *((float4*)((data0_16+alu0))) = (float4){(val0[0]+val1[0]),(val0[1]+val1[1]),(val0[2]+val1[2]),(val0[3]+val1[3])};
  }
}
'''

# https://mesozoic-egg.github.io/tinygrad-notes/20241231_intro.html