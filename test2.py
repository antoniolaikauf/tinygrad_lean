from tinygrad import nn, Tensor

class Model:
    def __init__(self):
        self.l1 = nn.Conv2d(1, 32, kernel_size=(3,3))
        self.l2 = nn.Conv2d(32, 64, kernel_size=(3,3))
        self.l3 = nn.Linear(64, 10)

    def __call__(self, x:Tensor) -> Tensor:
        x = self.l1(x).relu().max_pool2d((2,2))       # forma self.l1(x).relu() --> 1, 32, 10, 10             forma x = self.l1(x).relu().max_pool2d((2,2)) --> 1, 32, 5, 5
        x = self.l2(x).relu().max_pool2d((2,2))       # forma self.l2(x).relu() --> 1, 64, 3, 3               forma x = self.l2(x).relu().max_pool2d((2,2)) --> 1, 64, 1, 1
        return self.l3(x.flatten(1).dropout(0.5))     # flatten sulla seconda dimensione
    
x = Tensor.arange(144).reshape(1, 1, 12, 12)
print(x)
m = Model()
m(x)

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