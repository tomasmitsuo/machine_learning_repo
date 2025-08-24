import math
import data
import matplotlib.pyplot as plt


def calculate_euclidian_distance(item1, item2):
    if (len(item1) - len(item2) != 0):
        return "ERROR: SAMPLES DON'T HAVE THE SAME AMOUNT OF ATTRIBUTES"
    x = 0
    for i in range(len(item1) - 1):
        x += (item2[i] - item1[i])**2
    return math.sqrt(x)



def knn(training_set, test_sample, k, amostra): # It calculates all the distances between the test_sample and the training_set

    distances = []

    training_number = 1 # USED TO IDENTIFY WHAT'S THE TRAINING SAMPLE

    for training_item in training_set:
        distance = calculate_euclidian_distance(test_sample, training_item) # CALCULATES THE DISTANCE
        label = training_item[-1] # LAST INDEX IS THE LABEL
        distances.append((distance,label, training_number))
        training_number += 1
    
    distances.sort(key=lambda x:x[0])

    print(f"SAMPLE {amostra}:")
    for i in range(len(distances)):
        print(distances[i])
    
    result = 0
    for i in range(k):
        if distances[i][1] == 0: # BAD CASES
            result -= 1
        else:
            result += 1 # GOOD CASES

    if result > 0:    
        return 1
    return 0
    


def calculate_all_knn(training_set, test_set, k):
    i = 0
    result_samples = []
    
    amostra = 0
    for test_item in test_set:
        result_samples.append((test_item[0], test_item[1], knn(training_set, test_item, k, amostra)))
        i += 1
        amostra += 1

    return result_samples


# FUNCTIONS TO PLOT THE DATA AND VISUALIZE THE ALGORITHM (USE OF AI)

def _euclid(p, q):
    # p, q are (x, y, class) — use only features
    return math.hypot(p[0] - q[0], p[1] - q[1])

def plot_samples(training_set, labeled_test_set, k, graphic_name, link_neighbors=True):
    train_x0 = [x for x, y, c in training_set if c == 0]
    train_y0 = [y for x, y, c in training_set if c == 0]
    train_x1 = [x for x, y, c in training_set if c == 1]
    train_y1 = [y for x, y, c in training_set if c == 1]

    test_x0 = [x for x, y, c in labeled_test_set if c == 0]
    test_y0 = [y for x, y, c in labeled_test_set if c == 0]
    test_x1 = [x for x, y, c in labeled_test_set if c == 1]
    test_y1 = [y for x, y, c in labeled_test_set if c == 1]

    plt.figure(figsize=(8,6))

    # Training samples
    plt.scatter(train_x0, train_y0, c="blue", marker="o", label="Train - RUIM")
    plt.scatter(train_x1, train_y1, c="red",  marker="o", label="Train - BOM")

    # Test samples
    plt.scatter(test_x0,  test_y0,  c="blue", marker="^", edgecolors="black", label="Test - RUIM")
    plt.scatter(test_x1,  test_y1,  c="red",  marker="^", edgecolors="black", label="Test - BOM")

    # Connect each test point to its k nearest neighbors in training_set
    if link_neighbors and k > 0:
        for tx, ty, _ in labeled_test_set:
            # compute distances to every training point
            dists = [(_euclid((tx, ty, 0), tr), tr) for tr in training_set]
            dists.sort(key=lambda t: t[0])
            # take k nearest training points
            for _, (nx, ny, _) in dists[:k]:
                plt.plot([tx, nx], [ty, ny], linestyle="--", linewidth=1.0, alpha=0.4, color="gray")

    plt.xlabel("Total Sulfur Dioxide")
    plt.ylabel("Citric Acid")
    plt.title(f"Training vs Test Samples ---- k = {k}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(graphic_name)

# ================================================================================================================================================================


k = 7 # HIPERPARAMETRO do KNN


print("\n\nDADOS 2 FEATURES!!\n\n")
# 2 ATRIBUTOS NÃO NORMALIZADOS
labeled_testing_set = calculate_all_knn(data.training_data_2features, data.testing_data_2features, k)
plot_samples(data.training_data_2features, labeled_testing_set, k, f"sample_k_{k}.png")

print("\n\nDADOS 2 FEATURES NORMALIZADOS!!\n\n")
# 2 ATRIBUTOS NORMALIZADOS
labeled_testing_set = calculate_all_knn(data.normalized_training_data_2features, data.normalized_testing_data_2features, k)
plot_samples(data.normalized_training_data_2features, labeled_testing_set, k, f"normalized_sample_k_{k}.png")

# 11 ATRIBUTOS NÃO NORMALIZADOS

print("\n\nDADOS 11 FEATURES!\n\n")
calculate_all_knn(data.training_data_11features, data.testing_data_11features,k)

# 11 ATRIBUTOS NORMALIZADOS

print("\n\nDADOS 11 FEATURES NORMALIZADOS!\n\n")
calculate_all_knn(data.normalized_training_data_11features, data.normalized_testing_data_11features,k)

