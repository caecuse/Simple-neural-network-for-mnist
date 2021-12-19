# Sztuczne sieci neuronowe

## Wstęp
Celem programu jest implementacja perceptronu wielowarstwowego z wykorzystaniem optymalizacji gradientowej z algorytmem propagacji wstecznej, przy pomocy którego dokonywana jest klasyfikacja zbioru danych `MNIST` (http://yann.lecun.com/exdb/mnist/). Zbiór MNIST wczytywany jest z API `Keras` będącego częścią modułu `Tensorflow`
## Sposób działania
Na początku programu są pobierane i wstępnie przetwarzane dane ze zbioru MNIST. Następnie tworzona jest sieć neuronowa na podstawie podanych parametrów. Należą do nich:
* `shape` - tablica liczb całkowitych o długości > 1, oznaczająca kolejne rozmiary warstw ukrytych sieci
* `x_data`, `y_data` - dane uczące pobierane z API
* `activation` - sigmoidalna funkcja aktywacji neuronu, tutaj jest to funkcja $\frac{1}{1+exp(-x)}$
* `activation_prime` - pochodna funkcji activation, tu wynosząca $\frac{exp(x)}{(exp(x)+1)^2}$
* `loss` - funkcja straty, tu mająca  wzór$\frac{1}{N}\sum_{i=1}^{N} ||y_{i} - \hat{f}(x_{i};\theta)||^2$
* `loss_prime` - pochodna funkcji straty, tu mająca wzór $\frac{1}{N}\sum_{i=1}^{N} 2\cdot(y_{i} - \hat{f}(x_{i};\theta))$
* `epochs` - liczba epok, domyślnie równa 30
* `learning_rate` - parametr $\in (0,1)$ używany przy propagacji wstecznej z użyciem gradientu prostego, domyślnie równy 0,1
* `test_size` - rozmiar zbioru uczącego, domyślnie równy 1000

W tej implementacji sieci neuronowej każda warstwa reprezentowana jest przez dwie klasy: CLayer i ActiveLayer, gdzie CLayer reprezentuje statyczną część warstwy, zaś ActiveLayer część "myślącą". Obie klasy dziedziczą po tej samej klasie abstrakcyjnej Layer.
Po utworzeniu sieci i dodaniu do niej warstw uruchamiana jest pętla trenująca tyle razy ile wynosi wartość epochs. W jej wnętrzu wykonują się kolejno propagacja postępowa, ocena funkcji straty, oraz propagacja wsteczna. Po wyjściu z pętli sieć jest wytrenowana.
## Uruchamianie
Program może zostać uruchomiony na 2 sposoby
1. Bez żadnych argumentów wejściowych. Stworzona zostanie seria sieci dla parametrów test_size $\in$ {1000, 2000, 5000, 10000, 30000} oraz epochs $\in$ [200, 100, 50, 25], a następnie wykres trafności dla zbioru testowego mnist zawietającego 1000 osobników
2. Z dowolną liczbą l. całkowitych (które oznaczać będą wymiary dla kolejnych warstw ukrytych), oraz opcjonalnymi argumentami --test_size --epochs. Uruchomiona zostanie jedna sieć neuronowa dla podanych argumentów. Następnie wyświetlone zostaną wyniki oceny trafności w kolejnych epokach dla zbioru mnist oraz dla ręcznie narysowanych obrazów.
## Analiza wyników

## Dane o autorach
Program napisany został na ćwiczenia z przedmiotu Wstęp do Sztucznej
Inteligencji. Autorami programu są Jan Kędra (nr. ind. 310739) oraz Maksymilian Łazarski (nr ind. 310807)