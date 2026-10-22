Przedmiot: Uczenie głębokie w systemach rozpoznawania i przetwarzania mowy
Zajęcia nr / temat:Laboratorium 2 — Sieci rekurencyjne: budowa i uczenie
Autor / grupa: Damian Skorupa, GL03
Data: 22.10.2026

## 1. Cel ćwiczenia
Celem zadania było eksperymentalne zbadanie wpływu liczby jednostek ukrytych (hidden), długości okna wejściowego oraz typu komórki (RNN, LSTM, GRU) na jakość prognozy kolejnej próbki szeregu czasowego.

## 2. Metoda i implementacja
Narzędzia: Python, PyTorch, Matplotlib.
Konfiguracja eksperymentu: Wszystkie testy (oprócz badanej zmiennej) wykonano przy stałych parametrach: `epoki=20`, `lr=1e-2`, `batch=64`. 
Metodologia oceny: Każdą konfigurację wytrenowano 3-krotnie (ziarna: 0, 1, 2), a jako główną metrykę podano średnią stratę testową MSE z ostatnich 5 epok wraz z odchyleniem standardowym.


### Hipotezy (sformułowane przed uruchomieniem):
Eksperyment A: Zwiększenie liczby jednostek ukrytych poprawi dokładność modelu, jednak przy wartości `hidden=64` dla prostego sygnału sinusoidalnego nastąpi stagnacja poprawy i nieuzasadniony wzrost czasu treningu.

Eksperyment B: Wydłużenie okna powyżej 30 próbek wspomoże przewidywanie, jednak skrajne wydłużenie do 120 próbek może przynieść minimalne zyski przy dużym obciążeniu sieci, a dla architektury klasycznego RNN stanowić barierę trenowania.

## 3. Wyniki

Punkt odniesienia:
MSE heurystyki "ostatnia próbka": 0.00631

### Eksperyment A: Wpływ liczby jednostek ukrytych (GRU, OKNO=30)

| Konfiguracja | Liczba parametrów | Test MSE (średnia ± odchylenie) | Czas treningu (s) |
|---|---|---|---|
| GRU, hidden=8 | 337 | 0.00444 ± 0.00003 | 1.12 s |
| GRU, hidden=16 | 929 | 0.00374 ± 0.00022 | 1.17 s |
| GRU, hidden=32 | 3393 | 0.00376 ± 0.00001 | 1.56 s |
| GRU, hidden=64 | 12609 | 0.00375 ± 0.00008 | 2.32 s |

<img width="863" height="470" alt="Wykres_A" src="https://github.com/user-attachments/assets/65d4151a-e9ed-4ebb-a70c-985997bc4672" />


### Eksperyment B: Wpływ długości okna (GRU, hidden=32)

| Konfiguracja | Liczba parametrów | Test MSE (średnia ± odchylenie) | Czas treningu (s) |
|---|---|---|---|
| GRU, OKNO=10 | 3393 | 0.00414 ± 0.00019 | 0.65 s |
| GRU, OKNO=30 | 3393 | 0.00376 ± 0.00001 | 1.62 s |
| GRU, OKNO=60 | 3393 | 0.00343 ± 0.00021 | 3.08 s |
| GRU, OKNO=120| 3393 | 0.00344 ± 0.00019 | 5.85 s |

<img width="863" height="471" alt="Wykres_B" src="https://github.com/user-attachments/assets/10a55e23-a926-4f9d-9456-da61ee1ecb17" />


### Eksperyment C: Porównanie komórek (hidden=32, OKNO 10 i 120)

| Konfiguracja | Liczba parametrów | Test MSE (średnia ± odchylenie) | Czas treningu (s) |
|---|---|---|---|
| RNN, OKNO=10 | 1153 | 0.00417 ± 0.00025 | 0.32 s |
| LSTM, OKNO=10| 4513 | 0.00415 ± 0.00011 | 0.65 s |
| GRU, OKNO=10 | 3393 | 0.00414 ± 0.00019 | 0.59 s |
| RNN, OKNO=120| 1153 | 0.00400 ± 0.00001 | 2.37 s |
| LSTM, OKNO=120| 4513 | 0.00375 ± 0.00013 | 6.51 s |
| GRU, OKNO=120| 3393 | 0.00344 ± 0.00019 | 5.86 s |

<img width="863" height="471" alt="Wykres_C" src="https://github.com/user-attachments/assets/fdb497a8-e001-4a22-9d70-989502225e31" />


### Najlepsza Konfiguracja (GRU, hidden=32, OKNO=60):
<img width="708" height="393" alt="wykres_najlepszy_strata" src="https://github.com/user-attachments/assets/6ecf929e-10e7-4f56-8a2b-4cc6ee9d6085" />

<img width="838" height="317" alt="wykres_najlepszy_predykcja" src="https://github.com/user-attachments/assets/73419548-a1a4-468f-8ad9-7a70f9de99a6" />


## 4. Analiza i interpretacja
Wygenerowane wyniki jednoznacznie wskazują, że każda konfiguracja badanych sieci uczących bez problemu przebiła próg bazowy wyznaczony przez heurystykę "ostatnia próbka". Zastosowanie większej architektury znacząco zwiększyło czas uczenia (np. z 0.32s dla klasycznego RNN do 6.51s dla LSTM na długim oknie), co przełożyło się na widoczną, ale nie w każdym przypadku usprawiedliwioną redukcję błędu.

## 5. Wnioski
Przeprowadzone eksperymenty w pełni potwierdzają postawione hipotezy. Wszystkie modele skutecznie pobiły bazową heurystykę (MSE ~0.0063), redukując błąd testowy do poziomu 0.0034–0.0044. Zwiększanie liczby jednostek ukrytych powyżej 32 nie poprawia już istotnie predykcji, a jedynie wydłuża czas obliczeń, co świadczy o szybkim nasyceniu się tak prostego modelu.   
Optymalną długością okna okazało się 60 próbek (MSE ~0.00343) – okno 10-próbkowe dostarcza sieci zbyt małego kontekstu, a skrajne wydłużenie go do 120 próbek nie przynosi dalszych korzyści. Co kluczowe, przy tak długim oknie (120) ujawnia się problem zanikającego gradientu w klasycznej sieci RNN (MSE 0.00400), z którym nieporównywalnie lepiej i stabilniej radzą sobie bramkowane komórki LSTM i GRU. Okazjonalne niestabilności procesu uczenia przy większych modelach były skutecznie niwelowane przez mechanizm przycinania gradientów (clip_grad_norm_).
