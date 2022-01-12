#!/usr/bin/env python3

import argparse

import numpy as np
import pandas as pd
from keras.datasets import cifar10
import pickle
from tensorflow.keras import datasets, layers, models
# Custom Networks
import neural_networks as nn

# Helper functions
from differential_evolution import differential_evolution
import helper
import neural_networks as nn
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

class PixelAttacker:
    def __init__(self, models, testX, testY, class_names, dimensions=(28, 28)):
        # Load data and model
        self.models = models
        self.x_test=testX,
        self.y_test = testY
        self.class_names = class_names
        self.dimensions = dimensions

        network_stats, correct_imgs = helper.evaluate_models(self.models, self.x_test, self.y_test)
        self.correct_imgs = pd.DataFrame(correct_imgs, columns=['name', 'img', 'label', 'confidence', 'pred'])
        self.network_stats = pd.DataFrame(network_stats, columns=['name', 'accuracy', 'param_count'])

    def predict_classes(self, xs, img, target_class, model, minimize=True):
        # Perturb the image with the given pixel(s) x and get the prediction of the model
        print("in  predict classes")

        imgs_perturbed = helper.perturb_image(xs, img)
        #print(imgs_perturbed)
       # predictions = model.predict(imgs_perturbed)[:, target_class]
        predictions = model.predict(imgs_perturbed)[0][0][0]
        #print(predictions)
        # This function should always be minimized, so return its complement if needed
        return predictions if minimize else 1 - predictions

    def attack_success(self, x, img, target_class, model, targeted_attack=False, verbose=False):
        # Perturb the image with the given pixel(s) and get the prediction of the model
        attack_image = helper.perturb_image(x, img)

        confidence = model.predict(attack_image)[0]
        predicted_class = np.argmax(confidence)

        # If the prediction is what we want (misclassification or
        # targeted classification), return True
        if verbose:
            print('Confidence:', confidence[target_class])
        if ((targeted_attack and predicted_class == target_class) or
                (not targeted_attack and predicted_class != target_class)):
            return True

    def attack(self, img_id, model, target=None, pixel_count=1,
               maxiter=75, popsize=400, verbose=False, plot=False):
        # Change the target class based on whether this is a targeted attack or not
        targeted_attack = target is not None
        target_class = target if targeted_attack else self.y_test[img_id, 0]

        # Define bounds for a flat vector of x,y,r,g,b values
        # For more pixels, repeat this layout
        dim_x, dim_y = self.dimensions
        #bounds = [(0, dim_x), (0, dim_y), (0, 256), (0, 256), (0, 256)] * pixel_count
        bounds = [(0, 786), (0, 256), (0, 256), (0, 256)] * pixel_count

        # Population multiplier, in terms of the size of the perturbation vector x
        popmul = max(1, popsize // len(bounds))

        # Format the predict/callback functions for the differential evolution algorithm
        def predict_fn(xs):
            print("in predict fn")

            return self.predict_classes(xs, self.x_test[0][img_id], target_class, model, target is None)

        def callback_fn(x, convergence):
            return self.attack_success(x, self.x_test[0][img_id], target_class, model, targeted_attack, verbose)

        # Call Scipy's Implementation of Differential Evolution
        attack_result = differential_evolution(
            predict_fn, bounds, maxiter=maxiter, popsize=popmul,
            recombination=1, atol=-1, callback=callback_fn, polish=False)
        print("below pop mul")
        # Calculate some useful statistics to return from this function
        attack_image = helper.perturb_image(attack_result.x, self.x_test[0][img_id])[0]
        prior_probs = model.predict(np.array([self.x_test[0][img_id]]))[0]
        predicted_probs = model.predict(np.array([attack_image]))[0]
        predicted_class = np.argmax(predicted_probs)
        actual_class = np.argmax(self.y_test[img_id])
        print(actual_class)
        success = predicted_class != actual_class
        print(prior_probs)
        print(predicted_probs[0][0])
        print(actual_class)
        cdiff = prior_probs[actual_class] - predicted_probs[0][0][actual_class]

        # Show the best attempt at a solution (successful or not)
        if plot:
            helper.plot_image(attack_image, actual_class, self.class_names, predicted_class)

        return [model.name, pixel_count, img_id, actual_class, predicted_class, success, cdiff, prior_probs,
                predicted_probs, attack_result.x]

    def attack_all(self, models, samples=500, pixels=(1, 3, 5), targeted=False,
                   maxiter=75, popsize=400, verbose=False):
        results = []
        for model in models:
            model_results = []
            valid_imgs = self.correct_imgs[self.correct_imgs.name == model.name].img
            img_samples = np.random.choice(valid_imgs, samples)
            #print(img_samples)

            for pixel_count in pixels:
                for i, img in enumerate(img_samples):
                    print(model.name, '- image', img, '-', i + 1, '/', len(img_samples))
                    targets = [None] if not targeted else range(10)

                    for target in targets:
                        if targeted:
                            print('Attacking with target', self.class_names[target])
                            if target == self.y_test[img, 0]:
                                continue
                        #print(pixel_count)
                        result = self.attack(img, model, target, pixel_count,
                                             maxiter=maxiter, popsize=popsize,
                                             verbose=verbose)
                        model_results.append(result)

            results += model_results
            helper.checkpoint(results, targeted)
        return results


if __name__ == '__main__':
    model_defs = {
        #'cnn': nn.create_cnn(),
        #'rnn': nn.build_model(allow_cudnn_kernel=True),
        'mlp': nn.evaluate_mlp_model()
    }

    parser = argparse.ArgumentParser(description='Attack models on Cifar10')
    parser.add_argument('--model', nargs='+', choices=model_defs.keys(), default=model_defs.keys(),
                        help='Specify one or more models by name to evaluate.')
    parser.add_argument('--pixels', nargs='+', default=(1, 3, 5), type=int,
                        help='The number of pixels that can be perturbed.')
    parser.add_argument('--maxiter', default=75, type=int,
                        help='The maximum number of iterations in the differential evolution algorithm before giving up and failing the attack.')
    parser.add_argument('--popsize', default=400, type=int,
                        help='The number of adversarial images generated each iteration in the differential evolution algorithm. Increasing this number requires more computation.')
    parser.add_argument('--samples', default=500, type=int,
                        help='The number of image samples to attack. Images are sampled randomly from the dataset.')
    parser.add_argument('--targeted', action='store_true', help='Set this switch to test for targeted attacks.')
    parser.add_argument('--save', default='results.pkl', help='Save location for the results (pickle)')
    parser.add_argument('--verbose', action='store_true', help='Print out additional information every iteration.')

    args = parser.parse_args()
    feature_vector_length = 784
    # Load data and model
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    X_train = X_train.reshape(X_train.shape[0], feature_vector_length)
    X_test = X_test.reshape(X_test.shape[0], feature_vector_length)

    # Convert into greyscale
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    output_size=10

    # Convert target classes to categorical ones
    Y_train = to_categorical(Y_train, output_size)
    Y_test = to_categorical(Y_test, output_size)

    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    models = [model_defs[m] for m in args.model]

    attacker = PixelAttacker(models, X_test, Y_test, class_names)

    print('Starting attack')

    results = attacker.attack_all(models, samples=args.samples, pixels=args.pixels, targeted=args.targeted,
                                  maxiter=args.maxiter, popsize=args.popsize, verbose=args.verbose)

    columns = ['model', 'pixels', 'image', 'true', 'predicted', 'success', 'cdiff', 'prior_probs', 'predicted_probs',
               'perturbation']
    results_table = pd.DataFrame(results, columns=columns)

    print(results_table[['model', 'pixels', 'image', 'true', 'predicted', 'success']])
    with open('C:/Users/HP/PycharmProjects/FinalProject/Trust-Worthy-Machine-Learning/csv_results.csv', 'wb') as file:
        results_table.to_csv(file)

    print('Saving to', args.save)
    with open(args.save, 'wb') as file:
        pickle.dump(results, file)