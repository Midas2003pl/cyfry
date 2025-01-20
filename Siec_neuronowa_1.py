import numpy as np
import matplotlib.pyplot as plt
import utils

# Zapytaj użytkownika o wczytanie czy trening
choice = input("Czy chcesz wczytać wcześniej wytrenowane parametry? (tak/nie): ").strip().lower()

if choice == 'tak':
    # Wczytywanie wcześniej zapisanych parametrów
    weights_input_to_hidden = np.load("weights_input_to_hidden.npy")
    weights_hidden_to_output = np.load("weights_hidden_to_output.npy")
    bias_input_to_hidden = np.load("bias_input_to_hidden.npy")
    bias_hidden_to_output = np.load("bias_hidden_to_output.npy")

else:
    # Inicjalizacja wag i biasów
    hidden_neurons = 200  # 10, 20, 50, 100, 200

    weights_input_to_hidden = np.random.uniform(-0.5, 0.5, (hidden_neurons, 784))
    bias_input_to_hidden = np.zeros((hidden_neurons, 1))

    weights_hidden_to_output = np.random.uniform(-0.5, 0.5, (10, hidden_neurons))
    bias_hidden_to_output = np.zeros((10, 1))

    # Wczytanie datasetu
    images, labels = utils.load_dataset()

    epochs = 3
    learning_rate = 0.025

    #Otwórz plik CSV do zapisu wyników
    # results_file = open("experiment_results_epochs.csv", "w")
    # results_file.write("hidden_neurons,epochs,learning_rate,loss,accuracy\n")

    # Trening sieci
    for epoch in range(epochs):
        e_loss = 0
        e_correct = 0
        print(f"Epoch num.{epoch}")

        for image, label in zip(images, labels):
            image = np.reshape(image, (-1, 1))
            label = np.reshape(label, (-1, 1))

            # Forward propagation (warstwa ukryta)
            hidden_raw = bias_input_to_hidden + weights_input_to_hidden @ image
            hidden = 1 / (1 + np.exp(-hidden_raw)) # sigmoid (funkcja aktywacji)

            # Forward propagation (warstwa wyjściowa)
            output_raw = bias_hidden_to_output + weights_hidden_to_output @ hidden
            output = 1 / (1 + np.exp(-output_raw))

            # Obliczenie straty
            e_loss += 1 / len(output) * np.sum((output - label) ** 2, axis=0)
            e_correct += int(np.argmax(output) == np.argmax(label))

            # Backpropagation (warstwa wyjściowa)
            delta_output = output - label
            weights_hidden_to_output += -learning_rate * delta_output @ np.transpose(hidden)
            bias_hidden_to_output += -learning_rate * delta_output

            # Backpropagation (warstwa ukryta)
            delta_hidden = np.transpose(weights_hidden_to_output) @ delta_output * (hidden * (1 - hidden))
            weights_input_to_hidden += -learning_rate * delta_hidden @ np.transpose(image)
            bias_input_to_hidden += -learning_rate * delta_hidden
        
        # Po zakończeniu epoki zapisz stratę i dokładność
        avg_loss = e_loss[0] / images.shape[0]
        accuracy = e_correct / images.shape[0]
        # Dodaj wynik do pliku CSV
        # results_file.write(f"{hidden_neurons},{epochs},{learning_rate},{avg_loss},{accuracy}\n")

        # Wyświetl dane o procesie uczenia po każdej epoce
        print(f"Loss: {round((e_loss[0] / images.shape[0]) * 100, 3)}%")
        print(f"Accuracy: {round((e_correct / images.shape[0]) * 100, 3)}%")
    
    # results_file.close()

    # Zapisanie wytrenowanych parametrów do plików .npy
    np.save("weights_input_to_hidden.npy", weights_input_to_hidden)
    np.save("weights_hidden_to_output.npy", weights_hidden_to_output)
    np.save("bias_input_to_hidden.npy", bias_input_to_hidden)
    np.save("bias_hidden_to_output.npy", bias_hidden_to_output)

# Teraz użytkownik decyduje czy testujemy na pojedynczym obrazie, czy na zestawie testowym
test_choice = input("Czy chcesz przetestować pojedyńczy obraz (custom.jpg) czy testowy zestaw 10k obrazów? (wpisz 'o' lub 'z'): ").strip().lower()

if test_choice == 'z':
    # Wczytaj zestaw testowy (np. MNIST t10k)
    images_test, labels_test = utils.load_test_dataset()  # ta funkcja musi zostać zaimplementowana w 'utils'
    correct_predictions = 0

    for image, label in zip(images_test, labels_test):
        image = np.reshape(image, (-1, 1))

        # Forward propagation (warstwa ukryta)
        hidden_raw = bias_input_to_hidden + weights_input_to_hidden @ image
        hidden = 1 / (1 + np.exp(-hidden_raw)) # sigmoid

        # Forward propagation (warstwa wyjściowa)
        output_raw = bias_hidden_to_output + weights_hidden_to_output @ hidden
        output = 1 / (1 + np.exp(-output_raw))

        if np.argmax(output) == np.argmax(label):
            correct_predictions += 1

    accuracy = (correct_predictions / images_test.shape[0]) * 100

    results_file = open("experiment_results_neurons_hidden.csv", "a")

    results_file.write(f"{accuracy:.2f}\n")
    print(f"Dokładność na zestawie testowym: {accuracy:.2f}%")

    results_file.close()


else:
    # Test na pojedynczym obrazie custom.jpg
    test_image = plt.imread("custom.jpg", format="jpeg")

    # Konwersja do skali szarości + inwersja
    gray = lambda rgb : np.dot(rgb[... , :3] , [0.299 , 0.587, 0.114]) 
    test_image = 1 - (gray(test_image).astype("float32") / 255)

    # Reshape
    test_image = np.reshape(test_image, (test_image.shape[0] * test_image.shape[1]))

    # Predykcja na pojedynczym obrazie
    image = np.reshape(test_image, (-1, 1))

    # Forward propagation (warstwa ukryta)
    hidden_raw = bias_input_to_hidden + weights_input_to_hidden @ image
    hidden = 1 / (1 + np.exp(-hidden_raw)) # sigmoid

    # Forward propagation (warstwa wyjściowa)
    output_raw = bias_hidden_to_output + weights_hidden_to_output @ hidden
    output = 1 / (1 + np.exp(-output_raw))

    plt.imshow(test_image.reshape(28, 28), cmap="Greys")
    plt.title(f"The number on the picture is: {output.argmax()}")
    plt.show()