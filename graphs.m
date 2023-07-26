figure(1)
plot(sort(Real), "-")
hold on
plot(sort(Classified), "-")
legend("Real", "Classified")
xlabel("Number of points")
ylabel("Values")
grid on