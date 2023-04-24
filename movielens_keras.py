from __future__ import annotations
import pickle
import numpy as np
import tensorflow as tf

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot
import pandas as pd
from datetime import timedelta
from time import time


class L2(tf.keras.regularizers.Regularizer):
    """
    Un regularizador que aplica una penalización de regularización L2.
    La penalización de regularización L2 se calcula como:
    `loss = l2 * reduce_sum(square(x))`
    Argumentos:
        l2: Float; Factor de regularización L2.
    """

    def __init__(self, l2=0.01, **kwargs):
        l2 = kwargs.pop("l", l2)  # Backwards compatibility
        if kwargs:
            raise TypeError(f"Argument(s) not recognized: {kwargs}")

        l2 = 0.01 if l2 is None else l2

        self.l2 = tf.Variable(tf.keras.backend.cast_to_floatx(l2), trainable=False)

    def __call__(self, x):
        return self.l2 * tf.reduce_sum(tf.square(x))

    def get_config(self):
        return {
            "l2": float(tf.keras.backend.cast_to_floatx(self.l2.eval())),
        }


class UI_Callback(tf.keras.callbacks.Callback):
    def __init__(self, sender: Movielens_Learner) -> None:
        super().__init__()
        self.sender = sender

    def on_epoch_end(self, epoch, logs: dict | None = None):
        train_loss: float = logs["loss"]
        val_loss: float = logs["val_loss"]

        global iu_data
        iu_loss: float = np.nan

        if len(iu_data) > 0:
            iu_loss = self.model.evaluate(
                x=(iu_data["user"], iu_data["best_movie"], iu_data["worst_movie"]),
                y=np.ones(len(iu_data)),
                batch_size=len(iu_data),
                verbose=0,
            )

            print(f"IU loss: {iu_loss}")

        # emitimos la señal para pintar las películas
        self.sender.computed_embeddings.emit()

        # emitimos la señal para pintar el gráfico de errores
        self.sender.computed_avg_loss.emit(train_loss, val_loss, iu_loss)

        # emitimos para imprimir en consola los datos
        self.sender.mensaje.emit(
            f"Epoch: {epoch:4d} -> Training error: {train_loss:.6f} Test error: {val_loss:.6f}\n"
        )

        # emitimos para actualizar la barra de progreso
        self.sender.progreso.emit(epoch + 1)


class Movielens_Learner(QObject):
    # Señal que se emite cada vez que sacamos información durante el entrenamiento
    grafo_construido = pyqtSignal()
    grafo_eliminado = pyqtSignal()
    entrenamiento_finalizado = pyqtSignal()
    mensaje = pyqtSignal(str)
    progreso = pyqtSignal(int)
    # Señales que se emiten cada vez que sacamos información durante el entrenamiento
    computed_avg_loss = pyqtSignal(float, float, float)
    computed_embeddings = pyqtSignal()

    def __init__(
        self,
        num_users,
        num_movies,
        K=100,
        learning_rate=0.01,
        nu=0.000001,
        num_epochs=20,
        batch_size=250,
        random_seed=True,
        GPU=False,
    ) -> None:
        super().__init__()

        self.random_seed: bool = random_seed
        self.learning_rate: float = learning_rate
        self.nu: float = nu
        # el regularizador se inicializa en _init_graph, pero aquí lo declaro
        self.num_epochs: int = num_epochs
        self.batch_size: int = batch_size
        self.num_users: int = num_users
        self.num_movies: int = num_movies
        self.K: int = K

        # # La creación del grafo se deja para la primera vez que se intente entrenar el sistema
        self.the_model: tf.keras.Model | None = None
        self.model_for_prediction: tf.keras.Model | None = None

        self.use_GPU: bool = GPU

    def _init_graph(self) -> None:
        # Semilla de generador de aleatorios
        if self.random_seed:
            SEED = int(time())
        else:
            SEED = 2032

        # # tf.set_random_seed(SEED)
        # tf.compat.v1.random.set_random_seed(SEED)
        # random.seed(SEED)
        tf.random.set_seed(SEED)

        input_user = tf.keras.Input(shape=(None,), name="user")
        best_movie = tf.keras.Input(shape=(None,), name="best_movie")
        worst_movie = tf.keras.Input(shape=(None,), name="worst_movie")

        emb_reg = L2(l2=self.nu)
        W = tf.keras.layers.Embedding(
            self.num_users,
            self.K,
            name="W",
            embeddings_initializer=tf.initializers.GlorotUniform(),
            embeddings_regularizer=emb_reg,
        )
        V = tf.keras.layers.Embedding(
            self.num_movies,
            self.K,
            name="V",
            embeddings_initializer=tf.initializers.GlorotUniform(),
            embeddings_regularizer=emb_reg,
        )

        Wu = W(input_user)
        Vb = V(best_movie)
        Vw = V(worst_movie)

        f_best = tf.linalg.diag_part(
            tf.keras.layers.Dot(axes=2, normalize=False, name="WuVb")([Wu, Vb]),
            name="f_best",
        )
        f_worst = tf.linalg.diag_part(
            tf.keras.layers.Dot(axes=2, normalize=False, name="WuVw")([Wu, Vw]),
            name="f_worst",
        )
        the_output = tf.subtract(f_best, f_worst, name="salida")
        # outputs = [tf.linalg.diag_part(f_best), tf.linalg.diag_part(f_worst)]

        self.model_for_prediction = tf.keras.Model(
            inputs=[input_user, best_movie], outputs=f_best
        )

        self.the_model = tf.keras.Model(
            inputs=[input_user, best_movie, worst_movie], outputs=the_output
        )

        self.the_model.compile(
            # optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=self.learning_rate),
            loss=tf.keras.losses.Hinge(),
        )

        self.grafo_construido.emit()

    def set_params(self, **params) -> None:
        # K no va a poder ser modificada, sólo se permite desde la interfaz gráfica
        # cuando aún no se ha creado el modelo, así que sólo me tengo que preocupar
        # de propagar hacia el modelo existente los cambios en learning_rate y nu
        print(f"Params: {params}")
        for k, v in params.items():
            setattr(self, k, v)
            if self.the_model is not None:
                if k == "learning_rate":
                    self.the_model.optimizer.lr.assign(v)
                elif k == "nu":
                    # El regularizador es compartido por W y V así que basta con
                    # cambiarlo una vez
                    self.regularizer.l2.assign(v)

    def get_params(self) -> dict:
        hyperparams_list = [
            "K",
            "nu",
            "learning_rate",
            "batch_size",
            "num_epochs",
            "random_seed",
            "use_GPU",
            "num_users",
            "num_movies",
        ]

        params_actuales = {}
        for pname in hyperparams_list:
            params_actuales[pname] = getattr(self, pname)

        return params_actuales

    @property
    def regularizer(self) -> L2:
        return self.the_model.get_layer(name="W").embeddings_regularizer

    @property
    def V_weights(self) -> np.ndarray:
        return self.the_model.get_layer(name="V").get_weights()[0].transpose()

    @property
    def W_weights(self) -> np.ndarray:
        return self.the_model.get_layer(name="W").get_weights()[0].transpose()

    def reset_graph(self) -> None:
        self.the_model = None
        self.model_for_prediction = None
        self.grafo_eliminado.emit()

    def stop_fit(self) -> None:
        self.the_model.stop_training = True

    @pyqtSlot(pd.DataFrame, pd.DataFrame, name="fit")
    def fit(self, *args, **kwargs) -> None:
        if self.use_GPU:
            with tf.device("/device:GPU:0"):
                self.fit_aux(*args, **kwargs)
        else:
            with tf.device("/device:CPU:0"):
                self.fit_aux(*args, **kwargs)

    def fit_aux(self, data, test_data) -> None:
        if self.the_model == None:
            self._init_graph()

        global iu_data

        # Separamos los datos del usuario interactivo, si los hay, para ver
        # como se va comportando el modelo atendiendo únicamente a dicho usuario
        iu_data = data[data.user == self.num_users - 1]

        # Callback
        myCallback_obj: UI_Callback = UI_Callback(sender=self)
        self.the_model.fit(
            x=(data["user"], data["best_movie"], data["worst_movie"]),
            y=np.ones(shape=(len(data),)),
            batch_size=self.batch_size,
            validation_data=(
                (
                    test_data["user"],
                    test_data["best_movie"],
                    test_data["worst_movie"],
                ),
                np.ones(shape=(len(test_data),)),
            ),
            validation_batch_size=len(test_data),
            epochs=self.num_epochs,
            callbacks=myCallback_obj,
            shuffle=True,
            verbose=0,
        )

        self.entrenamiento_finalizado.emit()

    def predict(self, data):
        if self.the_model is None:
            raise RuntimeError("Modelo inexistente")

        predictions = self.model_for_prediction.predict([data.user, data.movie])
        df_result = pd.DataFrame(columns=["user", "movie", "predicted_score"])
        df_result.user = data.user
        df_result.movie = data.movie
        df_result.predicted_score = predictions
        return df_result

    def restore_model(self, path) -> None:
        params: dict = self.__load_params(path)
        self.set_params(**params)
        self._init_graph()
        # Y ahora tengo que ponerle la learning rate y el valor de nu, que no
        # se pudieron poner antes, por no existir aún el modelo
        self.set_params(**params)
        # Ahora cargamos los pesos
        self.the_model.load_weights(path)  # type: ignore

        # Y emitimos que ya hay modelo: se podrá grabar y exportar embeddings,
        # además de seguir entrenando
        self.grafo_construido.emit()

    def save(self, path: str) -> None:
        if self.the_model is not None:
            self.__save_topology(path)
            self.the_model.save_weights(path)

    def __save_topology(self, path: str) -> None:
        # Necesito salvar K, num_usuarios, num_peliculas y poco más...
        datos: dict = self.get_params()
        with open(path + "-myparams.pkl", "wb") as f:
            pickle.dump(datos, f)

    def __load_params(self, path: str) -> dict:
        with open(path + "-myparams.pkl", "rb") as f:
            datos: dict = pickle.load(f)
        return datos


def load_and_recode_pj(filename, new_user_codes=None, new_movie_codes=None) -> tuple:
    pj = pd.read_csv(filename, names=["user", "best_movie", "worst_movie"])

    # Recodificar usuarios y películas para que los índices comiencen en 0 y sean consecutivos
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
