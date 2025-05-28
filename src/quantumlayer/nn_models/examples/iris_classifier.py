from collections import defaultdict

import matplotlib.pyplot as plt
import perceval as pcvl
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm

from nn_models import QuantumLayer, OutputMappingStrategy, MLP, MLPConfig
from datasets import iris

train_features, train_labels, train_metadata = iris.get_data_train()
test_features, test_labels, test_metadata = iris.get_data_test()

# Convert data to PyTorch tensors
X_train = torch.FloatTensor(train_features)
y_train = torch.LongTensor(train_labels)
X_test = torch.FloatTensor(test_features)
y_test = torch.LongTensor(test_labels)


# Define model variants
def get_model_variants():
    """Define different variants for each model type"""
    # Define consistent colors for each model type
    MODEL_COLORS = {
        'MLP': '#1f77b4',  # Blue
        'LINEAR': '#2ca02c',  # Green
        'GROUPING': '#ff7f0e'  # Orange
    }

    # Define line styles for variants
    LINE_STYLES = ['--', '-', ':', '-.']

    variants = {
        'MLP': [
            {
                'name': 'MLP-Small',
                'config': MLPConfig(hidden_sizes=[8], dropout=0.1, activation='relu'),
                'color': MODEL_COLORS['MLP'],
                'linestyle': LINE_STYLES[0]
            },
            {
                'name': 'MLP-Medium',
                'config': MLPConfig(hidden_sizes=[16, 8], dropout=0.1, activation='relu', normalization='batch'),
                'color': MODEL_COLORS['MLP'],
                'linestyle': LINE_STYLES[1]
            }
        ],
        'LINEAR': [
            {
                'name': 'LINEAR-6modes',
                'config': {
                    'm': 6,
                    'output_mapping_strategy': OutputMappingStrategy.LINEAR
                },
                'color': MODEL_COLORS['LINEAR'],
                'linestyle': LINE_STYLES[0]
            },
            {
                'name': 'LINEAR-6modes-nobunching',
                'config': {
                    'm': 6,
                    'output_mapping_strategy': OutputMappingStrategy.LINEAR,
                    'no_bunching': True
                },
                'color': MODEL_COLORS['LINEAR'],
                'linestyle': LINE_STYLES[1]
            }
        ],
        'GROUPING': [
            {
                'name': 'GROUPING-6modes',
                'config': {
                    'm': 6,
                    'output_mapping_strategy': OutputMappingStrategy.GROUPING
                },
                'color': MODEL_COLORS['GROUPING'],
                'linestyle': LINE_STYLES[0]
            },
            {
                'name': 'GROUPING-6modes-nobunching',
                'config': {
                    'm': 6,
                    'output_mapping_strategy': OutputMappingStrategy.GROUPING,
                    'no_bunching': True
                },
                'color': MODEL_COLORS['GROUPING'],
                'linestyle': LINE_STYLES[1]
            }
        ]
    }
    return variants


def create_quantum_circuit(m):
    """Create quantum circuit with specified number of modes"""
    wl = pcvl.GenericInterferometer(m,
                                    lambda i: pcvl.BS() // pcvl.PS(pcvl.P(f"theta_li{i}")) // \
                                              pcvl.BS() // pcvl.PS(pcvl.P(f"theta_lo{i}")),
                                    shape=pcvl.InterferometerShape.RECTANGLE)

    c_var = pcvl.Circuit(m)
    for i in range(4):
        px = pcvl.P(f"px{i + 1}")
        c_var.add(i + (m - 4) // 2, pcvl.PS(px))

    wr = pcvl.GenericInterferometer(m,
                                    lambda i: pcvl.BS() // pcvl.PS(pcvl.P(f"theta_ri{i}")) // \
                                              pcvl.BS() // pcvl.PS(pcvl.P(f"theta_ro{i}")),
                                    shape=pcvl.InterferometerShape.RECTANGLE)

    c = pcvl.Circuit(m)
    c.add(0, wl, merge=True)
    c.add(0, c_var, merge=True)
    c.add(0, wr, merge=True)

    return c


def create_model(model_type, variant):
    """Create model instance based on type and variant"""
    if model_type == 'MLP':
        return MLP(input_size=4, output_size=3, config=variant['config'])
    else:
        m = variant['config']['m']
        no_bunching = variant['config'].get('no_bunching', False)
        c = create_quantum_circuit(m)
        thetas = [p.name for p in c.get_parameters() if not p.name.startswith("px")]
        return QuantumLayer(
            input_size=4, output_size=3,
            circuit=c, trainable_parameters=thetas,
            input_state= [1, 0] * (m // 2) + [0] * (m % 2),
            no_bunching=no_bunching,
            output_mapping_strategy=variant['config']['output_mapping_strategy']
        )

def count_parameters(model):
    """Count trainable parameters in a PyTorch model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_model(model, X_train, y_train, X_test, y_test, model_name, n_epochs=100, batch_size=32, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    losses = []
    train_accuracies = []
    test_accuracies = []

    model.train()

    pbar = tqdm(range(n_epochs), leave=False, desc=f"Training {model_name}")
    for epoch in pbar:
        permutation = torch.randperm(X_train.size()[0])
        total_loss = 0

        for i in range(0, X_train.size()[0], batch_size):
            indices = permutation[i:i + batch_size]
            batch_x, batch_y = X_train[indices], y_train[indices]

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / (X_train.size()[0] // batch_size)
        losses.append(avg_loss)
        pbar.set_description(f"Training {model_name} - Loss: {avg_loss:.4f}")

        # Evaluation
        model.eval()
        with torch.no_grad():
            train_outputs = model(X_train)
            train_preds = torch.argmax(train_outputs, dim=1).numpy()
            train_acc = accuracy_score(y_train.numpy(), train_preds)
            train_accuracies.append(train_acc)

            test_outputs = model(X_test)
            test_preds = torch.argmax(test_outputs, dim=1).numpy()
            test_acc = accuracy_score(y_test.numpy(), test_preds)
            test_accuracies.append(test_acc)

        model.train()

    # Generate final classification report
    model.eval()
    with torch.no_grad():
        final_test_outputs = model(X_test)
        final_test_preds = torch.argmax(final_test_outputs, dim=1).numpy()
        final_report = classification_report(y_test.numpy(), final_test_preds)

    return {
        'losses': losses,
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies,
        'final_test_acc': test_accuracies[-1],
        'classification_report': final_report
    }


def train_all_variants(X_train, y_train, X_test, y_test):
    """Train all model variants and return results"""
    variants = get_model_variants()
    all_results = defaultdict(dict)
    best_models = {}

    for model_type, model_variants in variants.items():
        print(f"\n\nTraining {model_type} variants:")
        best_acc = 0

        for variant in model_variants:
            model = create_model(model_type, variant)
            print(f"\nTraining {variant['name']}... ({count_parameters(model)} parameters)")

            results = train_model(model, X_train, y_train, X_test, y_test, variant['name'])
            results['model'] = model
            results['color'] = variant['color']
            results['linestyle'] = variant['linestyle']
            all_results[model_type][variant['name']] = results

            # Track best model for each type
            if results['final_test_acc'] > best_acc:
                best_acc = results['final_test_acc']
                best_models[model_type] = {
                    'name': variant['name'],
                    'model': model,
                    'results': results
                }

    return all_results, best_models


def plot_training_curves(all_results):
    """Plot training curves for all model variants"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))

    # Plot each metric
    for model_type, variants in all_results.items():
        for variant_name, results in variants.items():
            label = f"{variant_name}"
            color = results['color']
            linestyle = results['linestyle']

            ax1.plot(results['losses'], label=label, color=color, linestyle=linestyle, linewidth=2)
            ax2.plot(results['train_accuracies'], label=label, color=color, linestyle=linestyle, linewidth=2)
            ax3.plot(results['test_accuracies'], label=label, color=color, linestyle=linestyle, linewidth=2)

    # Customize plots
    for ax, title in zip([ax1, ax2, ax3], ['Training Loss', 'Training Accuracy', 'Test Accuracy']):
        ax.set_title(title, fontsize=12, pad=10)
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel(title.split()[-1], fontsize=10)
        ax.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_best_confusion_matrices(best_models, X_test, y_test):
    """Plot confusion matrices for the best model of each type"""
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    class_names = ['setosa', 'versicolor', 'virginica']

    for idx, (model_type, best) in enumerate(best_models.items()):
        model = best['model']
        model.eval()
        with torch.no_grad():
            outputs = model(X_test)
            predictions = torch.argmax(outputs, dim=1).numpy()

        cm = confusion_matrix(y_test.numpy(), predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names, ax=axes[idx])
        axes[idx].set_title(f'Best {model_type}\n({best["name"]})')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('True')
        plt.setp(axes[idx].get_xticklabels(), rotation=45)
        plt.setp(axes[idx].get_yticklabels(), rotation=45)

    plt.tight_layout()
    plt.show()


def print_comparison_results(all_results, best_models):
    """Print detailed comparison of all models and variants"""
    print("\n----- Model Comparison Results -----")

    # Print results for all variants
    print("\nAll Model Variants Results:")
    for model_type, variants in all_results.items():
        print(f"\n{model_type} Variants:")
        for variant_name, results in variants.items():
            print(f"\n{variant_name}:")
            print(f"Parameters: {count_parameters(results['model'])}")
            print(f"Final Test Accuracy: {results['final_test_acc']:.4f}")

    # Print best model results
    print("\nBest Models:")
    for model_type, best in best_models.items():
        print(f"\nBest {model_type} Model: {best['name']}")
        print(f"Final Test Accuracy: {best['results']['final_test_acc']:.4f}")
        print(f"Classification Report:\n{best['results']['classification_report']}")


# Train all variants
all_results, best_models = train_all_variants(X_train, y_train, X_test, y_test)

# Plot results
plot_training_curves(all_results)
plot_best_confusion_matrices(best_models, X_test, y_test)
print_comparison_results(all_results, best_models)