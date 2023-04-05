import numpy as np
import tensorflow as tf
from PyQt6.QtCore import QObject
import pandas as pd


class Movielens_PrefLoss(tf.keras.losses.Loss):
    # a ver si esto no hace falta poniendo ya la regularización en los embeddings...
    # def __init__(self, nu=1e-6, name="custom_prefloss"):
    #     super().__init__(name=name)
    #     self.regularization_factor = nu

    def call(self, best, worst):
        return tf.math.maximum(0.0, 1.0 - best + worst)


class Movielens_Model(tf.keras.Model):
    def __init__(self, num_users, num_movies, K, lr, nu):
        super().__init__()
        input_user = tf.keras.Input(shape=(None,))
        best_movie = tf.keras.Input(shape=(None,))
        worst_movie = tf.keras.Input(shape=(None,))
        emb_reg = tf.keras.regularizers.L2(l2=nu)
        W_embedding = tf.keras.layers.Embedding(
            num_users, K, name="W", embeddings_regularizer=emb_reg
        )
        V_embedding = tf.keras.layers.Embedding(
            num_movies, K, name="V", embeddings_regularizer=emb_reg
        )
        Wu = W_embedding(input_user)
        Vb = V_embedding(best_movie)
        Vw = V_embedding(worst_movie)

        f_best = tf.keras.layers.Dot(axes=1, normalize=False)([Wu, Vb])
        f_worst = tf.keras.layers.Dot(axes=1, normalize=False)([Wu, Vw])
        outputs = [f_best, f_worst]

        self.__the_model = tf.keras.Model(
            inputs=[input_user, best_movie, worst_movie], outputs=outputs
        )

    def call(self, inputs):
        return self.__the_model(inputs)


class Movielens_Learner(QObject):
    def __init__(
        self,
        num_users: int,
        num_movies: int,
        K: int = 100,
        learning_rate: float = 0.01,
        nu: float = 0.000001,
        num_epochs: int = 2,
        batch_size: int = 25,
        # batch_gen: str = "random",
        # drawevery: int = 2,
        random_seed: float | None = None,
        optimizer: str = "Adam",
        save_path: str = "movielens_model",
    ) -> None:
        super().__init__()
        self.__num_users = num_users
        self.__num_movies = num_movies
        self.__K = K
        self.__random_seed = random_seed
        self.__learning_rate = learning_rate
        self.__nu = nu
        self.__num_epochs = num_epochs
        self.__batch_size = batch_size

        self.model: tf.keras.Model = Movielens_Model(
            self.__num_users,
            self.__num_movies,
            self.__K,
            self.__learning_rate,
            self.__nu,
        )
        self.__loss: Movielens_PrefLoss = Movielens_PrefLoss(name="Movielens_Prefloss")
        self.__optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.__learning_rate
        )

        # No es necesario compilar el modelo, ya que utilizamos nuestro
        # propio bucle de entrenamiento.
        # self.model.compile(optimizer="Adam", loss=self.__loss)

    def fit(self, train_data, test_data):
        # Se pasan datos de test porque esta es una aplicación didáctica y queremos
        # ver qué va pasando con el conjunto de test a medida que vamos aprendiendo,
        # pero esto nunca se puede hacer en condiciones reales, ya que NUNCA se puede
        # usar el conjunto de test durante el entrenamiento

        # TODO voy aqui

        # Prepare the training dataset.
        train_dataset = tf.data.Dataset.from_tensor_slices(train_data)
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(self.__batch_size)

        # Prepare the validation dataset.
        test_dataset = tf.data.Dataset.from_tensor_slices(test_data)
        test_dataset = test_dataset.batch(self.__batch_size)

        # Bucle de entrenamiento (epochs)
        for epoch in range(self.__num_epochs):
            print(f"\nStart of epoch {epoch}")

            # Iterate over the batches of the dataset.
            for step, x_batch_train in enumerate(train_dataset):
                # Open a GradientTape to record the operations run
                # during the forward pass, which enables auto-differentiation.
                with tf.GradientTape() as tape:
                    # Run the forward pass of the layer.
                    # The operations that the layer applies
                    # to its inputs are going to be recorded
                    # on the GradientTape.
                    best, worst = self.model(
                        x_batch_train, training=True
                    )  # Logits for this minibatch

                    # Compute the loss value for this minibatch.
                    loss_value = self.__loss(best, worst)

                # Use the gradient tape to automatically retrieve
                # the gradients of the trainable variables with respect to the loss.
                grads = tape.gradient(loss_value, self.model.trainable_weights)

                # Run one step of gradient descent by updating
                # the value of the variables to minimize the loss.
                self.__optimizer.apply_gradients(
                    zip(grads, self.model.trainable_weights)
                )

                # Log every 200 batches.
                if step % 200 == 0:
                    print(
                        "Training loss (for one batch) at step %d: %.4f"
                        % (step, float(loss_value))
                    )
                    print("Seen so far: %s samples" % ((step + 1) * self.__batch_size))


def load_and_recode_pj(filename, new_user_codes=None, new_movie_codes=None) -> tuple:
    pj = pd.read_csv(filename, names=["user", "best_movie", "worst_movie"])

    # recodificar usuarios y películas para que los índices comiencen en 0 y sean consecutivos
    # Si ya se suministran los diccionarios de recodificación como parámetros, sólo se aplican, si no, se
    # calculan y luego se aplican.

    if new_user_codes == None:
        set_of_users = set(pj.user)
        new_user_codes = dict(zip(set_of_users, range(len(set_of_users))))
    if new_movie_codes == None:
        set_of_movies = set(pj.best_movie).union(set(pj.worst_movie))
        new_movie_codes = dict(zip(set_of_movies, range(len(set_of_movies))))

    users_recoded = [new_user_codes[i] for i in pj.user]
    pj.user = users_recoded
    best_movies_recoded = [new_movie_codes[i] for i in pj.best_movie]
    pj.best_movie = best_movies_recoded
    worst_movies_recoded = [new_movie_codes[i] for i in pj.worst_movie]
    pj.worst_movie = worst_movies_recoded

    return (pj, new_user_codes, new_movie_codes)
    # return (pj, len(set_of_users), len(set_of_movies))


if __name__ == "__main__":
    # Los conjuntos de entrenamiento y test van "atornillados" en el código, deben ser
    # pj_train.csv y pj_test.csv respectivamente
    train_pj, new_ucodes, new_mcodes = load_and_recode_pj("pj_train.csv")
    num_users = len(new_ucodes)
    num_movies = len(new_mcodes)
    # Los ejemplos de test se recodifican utilizando la codificación obtenida sobre los de entrenamiento, por
    # lo que es necesario que en el conjunto de entrenamiento aparezcan todos los usuarios y todas las películas
    # alguna vez, pero eso NO QUIERE DECIR que haya ejemplos de test en el conjunto de entrenamiento
    test_pj, _, _ = load_and_recode_pj("pj_test.csv", new_ucodes, new_mcodes)

    MV_learner: Movielens_Learner = Movielens_Learner(num_users, num_movies)
    MV_learner.fit(train_pj, test_pj)
