from tinygrad import Tensor

a = Tensor.empty(4, 4)
b = Tensor.empty(4, 4)
c = (a*b)
print(a.sum(0).tolist())

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


'''
kernel è una funzione in cuda marcata con  __global__ che è eseguita sulla GPU e chiamata dall'host CPU. non c'entra nulla con il kernel del sistema operativo che permette all'hardware di comunicare con il software
sono due cose diverse anche se hanno lo stesso nome.

nel codice sotto il kernel è lanciato da __launch_bounds__(4) che suggerisce l'uso di massimo 4 thread, i thread sono flussi di lavoro distinti all'interno di un singolo processo.
se una CPU ha un core allora non ci sarà parallelismo essendo che eseguirà un singolo thread ogni volta che starà allocando risorse a quel processo.
se invece ci sono più core allora si può fare parallelismo, con due core si possono eseguire due thread nello stesso processo essendo che i core sono come 'micro processori' con la propria cache e registri.
su sto pc ho 2 thread e 8 core quindi nell'esecuzione dei processi utilizzerò solo due core

int lidx0 = threadIdx.x; indica l'id del thread
float4 è un componente specifico di cuda, i componenti all'interno di foat4 sono x, y, z, w
make_float4 è un componente in cuda che permette di creare un foalt4

cuda   griglia --> blocchi --> thread  
ogni bloccco contiene la quantità di thread   

in cuda i thead eseguono lo stesso kernel (funzione) ma su dati differenti tra di loro, nella funzione con CUDA si può notare che non c'è nessun ciclo for per e cambiare l'ofset e accedere alla memoria
questo perchè quando il kernel viene eseguito vengono creati 4 thread che eseguono lo stesso kernel e in base all'id del thread che può essere 0, 1, 2, 3 l'ofset cambia e i thread accedono a diverse are di memoria.


#define INFINITY (__int_as_float(0x7f800000))
#define NAN (__int_as_float(0x7fffffff))
extern "C" __global__ void __launch_bounds__(4) E_4_4(float* data0, float* data1, float* data2) {
  int lidx0 = threadIdx.x; /* 4 */ numero di thread 
  int alu0 = (lidx0<<2);
  float4 val0 = *((float4*)((data1+alu0)));
  float4 val1 = *((float4*)((data2+alu0)));
  *((float4*)((data0+alu0))) = make_float4((val0.x+val1.x),(val0.y+val1.y),(val0.z+val1.z),(val0.w+val1.w));
}
'''

# https://mesozoic-egg.github.io/tinygrad-notes/20241231_intro.html


'''
per descrivere la sua computazione in tinygrad esiste la classe UOp e il modo in cui la computazione è rappresentata è An abstract syntax tree che consiste nel rappresentare il codice in una struttura ad albero
vedere https://en.wikipedia.org/wiki/Abstract_syntax_tree

class UOp:
  op: Ops
  dtype: dtypes
  src: tuple(UOp)
  arg: None
'''


# questo codice scritto in questo modo è 'scrittura a basso livello per tinygrad'
from tinygrad.uop.ops import UOp, Ops
from tinygrad import dtypes
from tinygrad.renderer.cstyle import CUDARenderer, MetalRenderer

# const = UOp(Ops.CONST, dtypes.float, arg=1.0)
# add = UOp(Ops.ADD, dtypes.float, src=(const, const), arg= None)
# print(add)
# print(CUDARenderer("sm_50").render([const, add]))

'''
la funzione render per generare codice 

Now if you check out the code generation implementation, you can see how it works. The render function iterates 
through the linearized UOp tree, and match each against the pattern specified in the PatternMatcher,
for each match it outputs the string, and the strings were combined at the end for the final rendered code.


tree

UOp(op=STORE, src=(
  UOp(op=DEFINE_GLOBAL),
  UOp(op=ADD, src=(
    UOp(op=LOAD, src=(
      UOp(op=DEFINE_GLOBAL)
    )),
    UOp(op=CONST, arg=1.0)
  ))
)

pattern

patterns = [
  (STORE, lambda uop: "="),
  (CONST, lambda uop: f" {uop.arg} "),
  (ADD, lambda uop: f" + "),
]

'''


# class UOp:
#   op: Ops
#   dtype: dtypes
#   src: tuple(UOp)
#   arg: None

# Metal_Renderer = MetalRenderer()
# CUDA_Renderer = CUDARenderer('sm_50')
# const = UOp(Ops.CONST, dtypes.float, arg=2.0) # variabile 1
# const1 = UOp(Ops.CONST, dtypes.float, arg=4.0) # variabile 2
# define_global = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=0.0) # area di memoria
# add = UOp(Ops.ADD, dtypes.float, arg=None, src=(const1, const)) # addizione tra due variabili
# store = UOp(Ops.STORE, dtypes.float, arg=None, src=(define_global, add)) # salvataggio della variabile nell'area di memoria

# uops = [const, define_global, const1,  add, store]
# rendered = CUDA_Renderer.render(uops)
# print(rendered)
# prova creazione di UOp codice 



# CUDA_Renderer = CUDARenderer('sm_50')
# const = UOp(Ops.CONST, dtypes.float, arg=2.0) # variabile 1
# const1 = UOp(Ops.CONST, dtypes.float, arg=4.0) # variabile 2
# define_global = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=0.0) # area di memoria
# xor = UOp(Ops.SUB, dtypes.float, arg=None, src=(const1, const)) # addizione tra due variabili
# store = UOp(Ops.STORE, dtypes.float, arg=None, src=(define_global, xor)) # salvataggio della variabile nell'area di memoria


# uops = [const, define_global, const1,  xor, store]
# rendered = CUDA_Renderer.render(uops)
# print(rendered)


# SHAPETRAKER permette di manipolare il tensore (per esempio la trasposizione) a quasi costo zero (cosi non da rallentare la computazione)
# le matrici nella memoria non sono rappresentate come le vediamo noi, ma come un array ad unica dimensione e per accedere agli elementi corretti di questo array salvato nella 
# memoria tinygrad usa "row * 2 + col" 
# shape, consiste nella forma della matrice, strides consiste in quanti passi fare quando si incrementa il valore, 2 farebbe riferimento alle row e 1 alle colonne 
# perchè per saltare da una row all'altra in una matrice di 2,2 bisogna saltare due elementi se fosse 2x3 bisognerebbe saltare 3 elementi, invece 1 fa riferimento alle colonne 
# perchè per spostarsi da colonna a colonna si salta un elemento 

'''
[
  0x00, 0x01
  0x02, 0x03
]

ma in memroia viene salvato [0x00, 0x01, 0x02, 0x03]
row * 2 + col sia row che col partono da 0 e quindi prendo il primo elemento 0x00, dopo row rimane 0 ma col va a 1 e quindi si prende, 0x01 row va a 2 e col va a 0 e si ottiene 0x02, e come ultimo row rimane a 2 e col è 1 e
si ottiene 0x03
'''

from tinygrad.shape.view import   View

a = View.create(shape=(2,2), strides=(2,1))

idx, valid = a.to_indexed_uops()

print(idx.render()) # questa è l'equazione usata per ottenere ogni singolo elemento dell'array nella memoria

# := indica che le due costanti sono le stesse instanze   
'''
# UOp(Ops.ADD, dtypes.int, arg=None, src=(
#   UOp(Ops.ADD, dtypes.int, arg=None, src=(
#     x1:=UOp(Ops.CONST, dtypes.int, arg=0, src=()),
#     UOp(Ops.MUL, dtypes.int, arg=None, src=(
#       UOp(Ops.RANGE, dtypes.int, arg=0, src=(
#          x1,
#         UOp(Ops.CONST, dtypes.int, arg=3, src=()),)),
#       x5:=UOp(Ops.CONST, dtypes.int, arg=2, src=()),)),)),

#   UOp(Ops.MUL, dtypes.int, arg=None, src=(
#     UOp(Ops.RANGE, dtypes.int, arg=1, src=(
#        x1,
#        x5,)),
#     UOp(Ops.CONST, dtypes.int, arg=1, src=()),)),))

((ridx0*2)+ridx1)
'''


CUDA_Renderer = CUDARenderer('sm_50')
x1 = UOp(Ops.CONST, dtypes.int, arg=0) 
x5 = UOp(Ops.CONST, dtypes.int, arg=2) 
x3 = UOp(Ops.CONST, dtypes.int, arg=3)

range1 = UOp(Ops.RANGE, dtypes.int, arg=(0,False), src=(x5,))
mul1 = UOp(Ops.MUL, dtypes.int, arg=None, src=(range1, x5))
add1 = UOp(Ops.ADD, dtypes.int, arg=None, src=(x1, mul1))

range2 = UOp(Ops.RANGE, dtypes.int, arg=(1,False), src=(x3,))
const1 = UOp(Ops.CONST, dtypes.int, arg=1) 
mul = UOp(Ops.MUL, dtypes.int, arg=None, src=(range2, const1))

add = UOp(Ops.ADD, dtypes.int, arg=None, src=(add1, mul))

uops = [x1, x5, x3, range1, mul1, add1, range2, const1, mul, add]
rendered = CUDA_Renderer.render(uops)
print(rendered)


a = View.create(shape=(3,2), strides=(2,1))
# quando si esegue una permute in tinygrad, l'array salvato nella memoria non cambia cosa che negli altri framework si 
# questo permette di essere più efficciente e risparmiare memoria 
'''
es 
[
  0x00, 0x01
  0x02, 0x03
  0x04, 0x05
]

new

[
  0x00, 0x02, 0x04
  0x01, 0x03, 0x05
]

e nella memoria si dovrebbe vedrebbe [0x00, 0x02, 0x04, 0x01, 0x03, 0x05] quindi non è contiguos (N.B i valori sarebbero address di memoria e non elementi)
ma in verità si vede ancora [0x00, 0x01, 0x02, 0x03, 0x04, 0x05], quando si modificano tensor si crea una view  all'interno di ShapeTracker  che è il componente responsabile di tracciare shape, strides e offset. 
Questa vista descrive un nuovo modo di interpretare e accedere ai dati esistenti, senza alcuna copia che cambia gli indici di accesso e non cambia la memoria sottostante 
'''

a.permute((1,0))
print(a.shape)
print(a.strides)

a.reshape((3,2))
print(a)



'''
in tinygrad qualsiasi operazioni con i tensor sono elementwise, o riduzione  cosi da avere un vantaggio 
di solito elementwise consiste in una operazione che non cambia il numero di elementi, invece una riduzione significa una 
riduzione di elementi lungo un asse 
'''

a = Tensor.empty(2,4)
b = Tensor.empty(4,3)
print(a.shape)
print(b.shape)

# print((a@b).tolist())


'''
convolution
'''

e = Tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]).reshape((1,1,4,4))
print(e.numpy())

weight = Tensor.ones((1, 1, 3, 3))
out = e.conv2d(weight)
print(out.numpy())

# per vedere il tensor muoversi sugli assi c'è un metodo chiamato _pool
# k_ si riferisce alla dimensione della maschera e stride fa riferimento a quanti elementi saltare
# dilation fa riferimento a quanti elementi prendere metre salta gli elementi in mezzo 


"""
a = Tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]).reshape((1,1,4,4))

[[[[ 0  1  2  3]
   [ 4  5  6  7]
   [ 8  9 10 11]
   [12 13 14 15]]]]


pooled = a._pool(k_=(2, 2), stride=1, dilation=2)



[[[[[[ 0  2]
     [ 8 10]]

    [[ 1  3]
     [ 9 11]]]


   [[[ 4  6]
     [12 14]]

    [[ 5  7]
     [13 15]]]]]]
"""

pooled = e._pool(k_=(3, 3), stride=1, dilation=1)
print(pooled.numpy())


'''

JIT

compilare il codice per la GPU richiede due passaggi:
primo è quello di compilare il kernel source code in byte code e questo avviene tramite il processo di rendering e viene generato un tree di UOp

extern "C" __global__ void __launch_bounds__(1) E_16(float* data0, float* data1, float* data2) {
  int gidx0 = blockIdx.x; /* 16 */
  float val0 = *(data1+gidx0);
  float val1 = *(data2+gidx0);
  *(data0+gidx0) = (val0+val1);
}

dopo aver fatto il rendering il codice sorgente viene compilato in byte code 
per la parte di compilazione viene il codice viene salvato tramite un metodo di cache, ogni kernel viene salvato con un proprio ckey
e se si ha gia generato ckey allora si ottiene il kernel se no si renderizza e si salva nella cache


tinyJIT sarebbe un involucro che serve per salvare i comnadi utilizzati dalla GPU cosi che si evita di ricreare il grafo di computazione e i comandi associati (vedi link https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g50d871e3bd06c1b835e52f2966ef366b)
in tinygrad quando un tensore viene 'realizzato' genera una sequenza di comandi specifici per le API che consistono in funzioni del venditore della GPU come caricare il codice compilato nella
memoria della GPU, trigger di esecuzione, senza JIT ogni chiamata ricrea questi comandi da zero  
'''


print("---------------------------------------------------\n\n\n")
from tinygrad import TinyJit
import time

weight = Tensor.empty(4, 4)

@TinyJit
def forward(x: Tensor):
    c = (x * weight).contiguous()
    c.sum(0).realize()


for i in range(4):
    start = time.time()
    x = Tensor.empty(4, 4)
    forward(x)
    end = time.time()
    print(f"Iteration {i} took {(end - start)*1000:.2f}ms")



'''
si nota che ci sono due kernel questo perchè lo scheduler di tinygrad non fusione tutte le due operazioni che si stanno facendo     c = (x * weight).contiguous()   c.sum(0).realize()
prima si fa  quella della creazione di c e dopo quella di sum

scheduled 2 kernels in 1.60 ms
*** CPU        7 E_4_4n2                                      arg  3 mem  0.00 GB tm     15.02us/     0.30ms (     0.00 GFLOPS    0.0|0.0     GB/s) ['contiguous', '__mul__', 'empty']
*** CPU        8 r_4_4n1                                      arg  2 mem  0.00 GB tm      9.65us/     0.31ms (     0.00 GFLOPS    0.0|0.0     GB/s) 
Iteration 0 took 9.08ms
scheduled 2 kernels in 1.85 ms
*** CPU        9 E_4_4n2                                      arg  3 mem  0.00 GB tm     24.57us/     0.34ms (     0.00 GFLOPS    0.0|0.0     GB/s) ['contiguous', '__mul__', 'empty']
*** CPU       10 r_4_4n1                                      arg  2 mem  0.00 GB tm      9.61us/     0.35ms (     0.00 GFLOPS    0.0|0.0     GB/s)
Iteration 1 took 3.22ms
scheduled 2 kernels in 2.88 ms
*** CPU       11 E_4_4n2                                      arg  3 mem  0.00 GB tm     22.22us/     0.37ms (     0.00 GFLOPS    0.0|0.0     GB/s) ['contiguous', '__mul__', 'empty']
*** CPU       12 r_4_4n1                                      arg  2 mem  0.00 GB tm     12.67us/     0.38ms (     0.00 GFLOPS    0.0|0.0     GB/s)
Iteration 2 took 5.31ms
scheduled 2 kernels in 1.97 ms
*** CPU       13 E_4_4n2                                      arg  3 mem  0.00 GB tm     12.86us/     0.40ms (     0.00 GFLOPS    0.0|0.0     GB/s) ['contiguous', '__mul__', 'empty']
*** CPU       14 r_4_4n1                                      arg  2 mem  0.00 GB tm      9.20us/     0.40ms (     0.00 GFLOPS    0.0|0.0     GB/s)
Iteration 3 took 3.34ms

si può notare che in ogni iterazione del ciclo vengono eseguiti due kernel, nella prima iterazione il codice viene generato e fatto in bytecode, il metodo di cache lo vedra per la prima volta e 
quindi lo salverà, all'interno della cache, nelle altre iterazioni in poi il codice verrà preso dalla cache e non generato ogni volta, ma le istruzioni dal grafo le creerà ogni volta 
'''


'''
con tinyJIT

scheduled 2 kernels in 1.67 ms
*** CPU        7 E_4_4n2                                      arg  3 mem  0.00 GB tm     19.47us/     0.23ms (     0.00 GFLOPS    0.0|0.0     GB/s) ['contiguous', '__mul__', 'empty']
*** CPU        8 r_4_4n1                                      arg  2 mem  0.00 GB tm     10.86us/     0.24ms (     0.00 GFLOPS    0.0|0.0     GB/s)
Iteration 0 took 11.97ms
scheduled 2 kernels in 1.66 ms
*** CPU        9 E_4_4n2                                      arg  3 mem  0.00 GB tm     26.88us/     0.27ms (     0.00 GFLOPS    0.0|0.0     GB/s) ['contiguous', '__mul__', 'empty']
*** CPU       10 r_4_4n1                                      arg  2 mem  0.00 GB tm      9.20us/     0.27ms (     0.00 GFLOPS    0.0|0.0     GB/s)
JIT captured 2 kernels with 0 inputs
Iteration 1 took 3.41ms
JIT GRAPHing batch with 2 kernels on device <tinygrad.runtime.ops_cpu.CPUDevice object at 0x7c331c98fa60>
*** CPU       11 <batched 2>                                  arg  0 mem  0.00 GB tm     92.61us/     0.37ms (     0.00 GFLOPS    0.0|0.0     GB/s)
Iteration 2 took 1.61ms
*** CPU       12 <batched 2>                                  arg  0 mem  0.00 GB tm     86.53us/     0.45ms (     0.00 GFLOPS    0.0|0.0     GB/s)
Iteration 3 took 0.73ms


si può vedere che durante la compilazione del codice c'è voluto di più (11.97ms), questo perchè deve essere creato il bytecode e dopo la pipiline/grafo
neella seconda iterazione si salta il bytecode ma si ripete lo stesso la pipeline/grafo, ma sta volta tinyJIT cattura il pipeline/grafo quindi i comandi API/i kernel


JIT captured 2 kernels with 0 inputs
Iteration 1 took 3.41ms
JIT GRAPHing batch with 2 kernels on device <tinygrad.runtime.ops_cpu.CPUDevice object at 0x7c331c98fa60>
*** CPU       11 <batched 2>                                  arg  0 mem  0.00 GB tm     92.61us/     0.37ms (     0.00 GFLOPS    0.0|0.0     GB/s)
qua il <batched 2> indica i due kernel che venivano fatti nella prima e seconda iterwazione

ma dalla iterazione due e tre la velocità aumenta di molto

non si possono avere due jit nested, nell'esempio di prima si fa il JIT sul forward  se si facesse anche il JIT sul training loop darebbe errore

in tinygrad se si usasse una variabile come appoggio es. una variabile che viene usata per scambiare il contenuto di due variabili 
temp = a
a = b
b = temp

la memoria della variabile temporanea non verrebbe salvata ogni volta nel forward
ma se si usasse tinyJIT invece si salverebbe costantemente, senza ogni volta allocare la memoria ad ogni chiamata della funzione 
'''


import math

@TinyJit
def replace_if_zero(tokens: Tensor):
  tokens = (tokens == 0).where(-math.inf, tokens).contiguous()
  _ = (tokens == 0).all()
  return tokens, _ 

ctx = Tensor([
  [0, 0]
])

_a = [
    [1, 7],
    [2, 0],
    [3, 2],
    [4, 0],
]

for i in range(0, len(_a)):
  a = Tensor(_a[i]).reshape((1, -1))
  a, _ = replace_if_zero(a)
  print('qua', _.numpy())
  ctx = ctx.cat(a, dim=0)

print(ctx.numpy())



k = Tensor.empty(4)
k = k.sum(0)
print(k.numpy())

'''
features VIZ permette di vedere tutte i kernel e istruzioni in una pagina web

a destra della pagina ti darà il codice renderizzato e a sinistra i kernel 
in centro si trova un AST albero sintattico astratto che rappresenta la struttura di un albero del codice sorgente 
SINK signifca che la computazione è finita 
STORE significa che il risultato della computazione viene salvato, per salvar eun risultato abbiamo bisogno di 3 componenti dove salvarlo, cosa salvare e come salvarlo 
DEFINE_GLOBAL con argomento 0 significa il DOVE e nel nostro codice sarebbe float* restrict data0_1
il VIEW sarebbe il come che nel codice sarebbe *(data0_1+0)
il come sarebbe quello che viene dopo il segno '=' e in VIZ sarebbe REDUCE_AXIS che ha un argomento 0 questo perchè stiamo facendo la somma sull asse 0
dopo ha il op ADD questo perchè stiamo facendo la somma
e l'input da prendere è preso da DEFINE_GLOBAL con valore 1


nella parte a destra in alto c'è la chiamata della funzione per il grafo
return graph_rewrite(sink, self.pm, ctx=self.ctx(sink) if self.ctx is not None else None, name=self.name, bottom_up=self.bottom_up)
subito sotto abbiamo il testo del codice quando viene realizzato 
'''
