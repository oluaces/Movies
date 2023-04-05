from operator import le
from re import L
import numpy as np
import tensorflow as tf

# from PyQt6.QtCore import QObject
import pandas as pd

from movielens_tf2_qt import Movielens_Learner


# class Movielens_PrefLoss(tf.keras.losses.Loss):
#     def call(self, best, worst):
#         return tf.math.maximum(0.0, 1.0 - best + worst)


class Movielens_Model(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
     

    def create_model(self,
        num_users: int,
        num_movies: int,
        K: int = 100,
        learning_rate: float = 1e-2,
        nu: float = 1e-4,
    ) -> tf.keras.Model:
        input_user = tf.keras.Input(shape=(None,), name="usuario")
        best_movie = tf.keras.Input(shape=(None,), name="mejor película")
        worst_movie = tf.keras.Input(shape=(None,), name="peor película")
        emb_reg = tf.keras.regularizers.L2(l2=nu)
        W_embedding = tf.keras.layers.Embedding(
            num_users,
            K,
            name="W",
            embeddings_initializer=tf.keras.initializers.GlorotUniform(),
            embeddings_regularizer=emb_reg,
        )
        V_embedding = tf.keras.layers.Embedding(
            num_movies,
            K,
            name="V",
            embeddings_initializer=tf.keras.initializers.GlorotUniform(),
            embeddings_regularizer=emb_reg,
        )
        Wu = W_embedding(input_user)
        Vb = V_embedding(best_movie)
        Vw = V_embedding(worst_movie)

        f_best = tf.keras.layers.Dot(axes=1, normalize=False, name="WuVb")([Wu, Vb])
        f_worst = tf.keras.layers.Dot(axes=1, normalize=False, name="WuVw")([Wu, Vw])
        outputs = [tf.linalg.diag_part(f_best), tf.linalg.diag_part(f_worst)]

        the_model = Movielens_Model(
            inputs=[input_user, best_movie, worst_movie], outputs=outputs
        )
        the_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            # loss=Movielens_PrefLoss,
        )
        return the_model

    # def kk_train_step(self, data):
    #     print(f"\n\n{data}\n"+"-"*60)
    #     u, b, w = data[:, 0], data[:, 1], data[:, 2]
    #     output = self([u, b, w], training=True)
    #     print(output)
    #     return {"loss": 0.0}

    def train_step(self, data):
        # We can change the default convention for parameters (tuple x, y and weights)
        # and use any data we want.
        u, b, w = data[:, 0:1], data[:, 1:2], data[:, 2:]

        with tf.GradientTape() as tape:
            y_best, y_worst = self([u, b, w], training=True)
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            # loss = self.compiled_loss(y_best, y_worst)
            loss = tf.math.maximum(0.0, 1.0 - y_best + y_worst)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
         # Compute our own metrics
        self.loss_tracker.update_state(loss)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    @property
    def metrics(self):
        return [self.loss_tracker]

    # def test_step(self, data):
    #     print("\n\nTEST STEP\n\n")
    #     u, b, w = data[:, 0], data[:, 1], data[:, 2]
    #     y_best, y_worst = self([u, b, w], training=False)

    #     self.compiled_loss(y_best, y_worst, regularization_losses=self.losses)
    #     self.compiled_metrics.update_state(y_best, y_worst)

    #     return {m.name: m.result() for m in self.metrics}





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
    # Desabilitar GPU, en Mac M1/M2 no va fino :(
    tf.config.set_visible_devices([], 'GPU')

    # Los conjuntos de entrenamiento y test van "atornillados" en el código, deben ser
    # pj_train.csv y pj_test.csv respectivamente
    train_pj, new_ucodes, new_mcodes = load_and_recode_pj("pj_train.csv")
    num_users = len(new_ucodes)
    num_movies = len(new_mcodes)
    # Los ejemplos de test se recodifican utilizando la codificación obtenida sobre los de entrenamiento, por
    # lo que es necesario que en el conjunto de entrenamiento aparezcan todos los usuarios y todas las películas
    # alguna vez, pero eso NO QUIERE DECIR que haya ejemplos de test en el conjunto de entrenamiento
    test_pj, _, _ = load_and_recode_pj("pj_test.csv", new_ucodes, new_mcodes)


    batch_size = 250
    # Prepare the training dataset.
    train_dataset = tf.data.Dataset.from_tensor_slices(train_pj)
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

    # Prepare the validation dataset.
    test_dataset = tf.data.Dataset.from_tensor_slices(test_pj)
    test_dataset = test_dataset.batch(batch_size)

    MV_learner: Movielens_Model = Movielens_Model().create_model(num_users, num_movies, learning_rate=0.001, nu=1e-4)
    MV_learner.summary()
    tf.keras.utils.plot_model(MV_learner, to_file="kk.png")
    # print(list(train_dataset.as_numpy_iterator()))
    MV_learner.fit(train_dataset, epochs=30)
