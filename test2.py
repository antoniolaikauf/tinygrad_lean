from tinygrad import nn, Tensor
from tinygrad.nn.datasets import mnist 
class Model:
    def __init__(self):
        self.l1 = nn.Conv2d(1, 32, kernel_size=(3,3)) # qua la maschera kernel si sposta di stride, non inizializzato sarebbe 1, se noi mettessimo 2 allora si sposterebbe di due e non otterremmo # forma self.l1(x).relu() --> 1, 32, 10, 10 ma # forma self.l1(x).relu() --> 1, 32, 5, 5
        self.l2 = nn.Conv2d(32, 64, kernel_size=(3,3))
        self.l3 = nn.Linear(1600, 10)

    def __call__(self, x:Tensor) -> Tensor:
        x = self.l1(x).relu().max_pool2d((2,2))       # forma self.l1(x).relu() --> 1, 32, 10, 10             forma x = self.l1(x).relu().max_pool2d((2,2)) --> 1, 32, 5, 5
        # print(x.size())
        x = self.l2(x).relu().max_pool2d((2,2))       # forma self.l2(x).relu() --> 1, 64, 3, 3               forma x = self.l2(x).relu().max_pool2d((2,2)) --> 1, 64, 1, 1
        # print('qua',x.size(), x.flatten(1).size())
        return self.l3(x.flatten(1).dropout(0.5))     # flatten sulla seconda dimensione
'''
x = Tensor.arange(144).reshape(1, 1, 12, 12)
print(x)
m = Model()
m(x)
'''

# max_pool2d è una finestra di tot dimensioni che ottiene l'elemento più grande di quella finestra di dimensione 
# [1, 2, 3, 4]
# [5, 6, 7, 8]
# [9, 10, 11, 12]
# [13, 14, 15, 16]

# Regione 1: [1, 2, 5, 6] → massimo = 6
# Regione 2: [3, 4, 7, 8] → massimo = 8
# Regione 3: [9, 10, 13, 14] → massimo = 14
# Regione 4: [11, 12, 15, 16] → massimo = 16

# [6, 8]
# [14, 16]

# N.B  la dimensione del kernel deve essere sempre maggiore di quella dell'input sia nella conv2d che nella max_conv2d 

# dropout permette di azzerare dei neuroni e avviene nel seguente modo: se si inserisce un dropout di 0.5 allora si spegneranno il 50% pdi neuroni, torch crea una
# maschera con una probabilità distribuita di Bernoulli (è una probabilità di solo due valori 0, 1 detti anche fallimento e successo) [0, 1, 1, 0] e questa maschera viene moltiplicata ai neuroni 


X_train, Y_train, X_test, Y_test = mnist()

m = Model()
# argmax ritorna l'indice per ogni row e ritorna un tensor di dimensioni (10000,) che dopo viene comparato con quello di Y_test per creare un tensor di valori booleani e alla fine viene fatta la media
# per capire quante ne ha fatte corrette
acc = (m(X_test).argmax(axis=1) == Y_test).mean()

optim = nn.optim.Adam(nn.state.get_parameters(m))
batch_size = 128

# training
def step():
    # attivazione allenamento e quindi fa attiva il dropout 
    Tensor.training = True
    samples = Tensor.randint(batch_size, high=X_train.shape[0])
    X, Y = X_train[samples], Y_train[samples]
    # azzeramento dei gradianti
    optim.zero_grad()
    # calcolo loss 
    loss = m(X).sparse_categorical_crossentropy(Y).backward()
    #aggiornamento dei gradianti
    optim.step()
    return loss

for stepId in range(7000):
    loss = step()
    # test ogni 100 iterazioni
    if stepId % 100 == 0:
        Tensor.training = False
        acc = (m(X_test).argmax(axis=1) == Y_test).mean().item()
        print(f"step {stepId:4d}, loss {loss.item():.2f}, acc {acc*100.:.2f}%")