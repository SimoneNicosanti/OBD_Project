figure(1)
plot((Real-Classified).^2)
title("Errore Quadratico del Test Set")
legend("Errore Quadratico")
xlabel("Campioni")
ylabel("Errore Quadratico")
grid on

pd = fitdist(Real-Classified,'Normal');
h = kstest(Real-Classified);

figure(2)
plot(pd)
title("Residui del Test Set")
legend("Residui", "Gaussiana")
xlabel("Errore")
ylabel("Numero di Campioni")
grid on

disp(pd)
disp(h)