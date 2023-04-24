from __future__ import annotations
import pickle
import numpy as np
import tensorflow as tf

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot
import pandas as pd
from datetime import timedelta
from time import time
import math

# from tensorflow.python.util.tf_export import keras_export


# class Movielens_PrefLoss(tf.keras.losses.Loss):
#     def call(self, best, worst):
#         return tf.math.maximum(0.0, 1.0 - best + worst)


# class Movielens_Model(tf.keras.Model):
#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         self.loss_tracker = tf.keras.metrics.Mean(name="pref. loss")

#     def train_step(self, data):
#         u, b, w = data[:, 0:1], data[:, 1:2], data[:, 2:3]

#         with tf.GradientTape() as tape:
#             y_best, y_worst = self([u, b, w], training=True)  # type: ignore
#             # loss = self.compiled_loss(y_best, y_worst)
#             loss = tf.math.maximum(0.0, 1.0 - y_best + y_worst)

#         # Calcular gradientes
#         trainable_vars = self.trainable_variables
#         gradients = tape.gradient(loss, trainable_vars)
#         # Actualizar los pesos
#         self.optimizer.apply_gradients(zip(gradients, trainable_vars))
#         # Calcular nuestra propia métrica
#         self.loss_tracker.update_state(loss)

#         # Retornar un dict mapeando los nombres de las métricas con su valores
#         return {m.name: m.result() for m in self.metrics}

#     @property
#     def metrics(self):
#         return [self.loss_tracker]

#     def test_step(self, data):
#         u, b, w = data[:, 0:1], data[:, 1:2], data[:, 2:3]
#         y_best, y_worst = self([u, b, w], training=False)  # type: ignore
#         loss = tf.math.maximum(0.0, 1.0 - y_best + y_worst)
#         self.loss_tracker.update_state(loss)

#         # Retornar un dict mapeando los nombres de las métricas con su valores
#         return {m.name: m.result() for m in self.metrics}


# @keras_export("keras.regularizers.L2", "keras.regularizers.l2")
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
            # "name": self.__class__.__name__,
            "l2": float(tf.keras.backend.cast_to_floatx(self.l2.eval())),
        }


class MyCallback(tf.keras.callbacks.Callback):
    def __init__(self, sender: Movielens_Learner) -> None:
        super().__init__()
        self.sender = sender

    def on_epoch_end(self, epoch, logs: dict | None = None):
        # print(f"Lo que llega a on_epoch_end: epoch={epoch}, logs={logs}")
        train_loss: float = logs["loss"]
        val_loss: float = logs["val_loss"]

        # print(f"Train loss: {train_loss}")
        # print(f"Test loss: {val_loss}")

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
    # computed_avg_loss = pyqtSignal(int, float, float, float)
    # computed_embeddings = pyqtSignal(np.ndarray, np.ndarray)
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
        # batch_gen="random",
        # drawevery=2,
        random_seed=True,
        # optimizer="Adam",
        # save_path="movielens_model",
        GPU=False,
    ) -> None:
        super().__init__()

        self.random_seed: bool = random_seed
        self.learning_rate: float = learning_rate
        self.nu: float = nu
        # el regularizador se inicializa en _init_graph, pero aquí lo declaro
        self.emb_reg: L2
        self.num_epochs: int = num_epochs
        self.batch_size: int = batch_size
        # self.batch_gen = batch_gen
        self.num_users: int = num_users
        self.num_movies: int = num_movies
        self.K: int = K
        # self.optimizer = optimizer
        # self.save_path = save_path

        # flag que se usa para interrumpir un entrenamiento
        # self.parar_entrenamiento = False
        # # Lista que contiene los steps y errores medios
        # self.global_step = self.average_loss = 0
        # self.drawevery = drawevery  # cada 'drawevery' iteraciones se dibuja

        # # La creación del grafo se deja para la primera vez que se intente entrenar el sistema
        # self.the_model: tf.keras.Model | None = None
        # self.model_for_prediction: tf.keras.Model | None = None
        self.the_model: tf.keras.Model | None = None
        self.model_for_prediction: tf.keras.Model | None = None

        self.use_GPU: bool = GPU

    def _init_graph(self) -> None:
        # Semilla de generador de aleatorios
        if self.random_seed:
            SEED = int(time())
        else:
            SEED = 2032

        # Desabilitar GPU?
        # physical_devices = tf.config.list_physical_devices("GPU")
        # if self.use_GPU:
        #     tf.config.set_visible_devices(physical_devices, "GPU")
        # else:
        #     tf.config.set_visible_devices([], "GPU")

        print("Physical devices:\n", tf.config.list_physical_devices())
        print("Logical devices:\n", tf.config.list_logical_devices())

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
            # "drawevery",
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
        # myCallback_obj: MyCallback = MyCallback(
        #     señal_errores=self.computed_avg_loss,
        #     señal_embeddings=self.computed_embeddings,
        # )
        myCallback_obj: MyCallback = MyCallback(sender=self)
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

    """
    def fit_aux(
        self,
        data: pd.DataFrame,
        test_data: pd.DataFrame,
        *args,
        **kwargs,
    ) -> None:
        if self.the_model is None:
            self._init_graph()

        if self.the_model is None:
            raise RuntimeError("Modelo no creado")

        # Preparamos el dataset de entrenamiento.
        training_data: tf.data.Dataset = tf.data.Dataset.from_tensor_slices(data)
        num_examples: int = training_data.cardinality()
        training_data = training_data.shuffle(buffer_size=num_examples).batch(
            batch_size=self.batch_size
        )

        # Y el de validación, aquí usamos el de test porque es una aplicación didáctica pero esto
        # NO SE PUEDE HACER EN UN EXPERIMENTO REAL: EL CONJUNTO DE TEST NUNCA SE PUEDE VER DURANTE
        # EL ENTRENAMIENTO.
        validation_data: tf.data.Dataset = tf.data.Dataset.from_tensor_slices(test_data)
        validation_data = validation_data.batch(
            batch_size=validation_data.cardinality()
        )

        # También separamos los datos del usuario interactivo, si los hay, para ver
        # como se va comportando el modelo atendiendo únicamente a dicho usuario
        data_interactive_user = data[data.user == self.num_users - 1]

        iu_data: tf.data.Dataset | None = None
        if len(data_interactive_user) > 0:
            iu_data = tf.data.Dataset.from_tensor_slices(data_interactive_user)
            iu_data = iu_data.batch(batch_size=iu_data.cardinality())

        # Aquí vamos de batch en batch haciendo entrenamiento y test, para ver como evoluciona
        # el error en ambos conjuntos. Esto es con caracter didáctico, no se puede hacer así
        # en la resolución de un problema real.
        self.mensaje.emit("\n**** COMIENZA EL ENTRENAMIENTO ****\n")

        t1: float = time()
        # average_loss = 0
        steps_between_updates: int = 0
        self.parar_entrenamiento = False
        steps_per_epoch = int(math.ceil(num_examples / self.batch_size))

        contador: int = 0
        t_medio: float = 0

        for epoch in range(self.num_epochs):
            # Iterar sobre los batches del conjunto de entrenamiento
            metric_values: dict = {}
            for step, x_batch_train in enumerate(training_data):
                # print(f"Training step {step}")
                # batch_size: int = x_batch_train.shape[0]

                t_antes: float = time()
                metrics_values = self.the_model.train_step(x_batch_train)
                t_medio += time() - t_antes
                contador += 1

                self.global_step += 1
                steps_between_updates += 1

                t2: float = time()
                última_iteración: bool = self.parar_entrenamiento or (
                    epoch == self.num_epochs - 1 and step == steps_per_epoch - 1
                )
                if (t2 - t1) >= self.drawevery or última_iteración:
                    # Al ejecutar la última iteración dejamos actualizados los gráficos y los embeddings
                    avg_loss: float = metrics_values["pref. loss"].numpy()
                    # The average loss is an estimate of the loss over the last 'drawevery' batches.
                    msg: str = f"Epoch {epoch+1:3d}, step {step} ->\tError medio (entrenamiento): {avg_loss:.6f}\n"

                    # Reset de las métricas de entrenamiento al final de paso de entrenamiento,
                    # necesario antes de contabilizar la misma métrica para el test y el usuario interactivo
                    for m in self.the_model.metrics:
                        m.reset_states()

                    # Ahora miramos qué tal vamos en el conjunto de test (esto se hace porque esta es una aplicación
                    # didáctica, no podríamos hacerlo en condiciones reales, ya que no se puede utilizar
                    #  NUNCA el conjunto de test para entrenar
                    val_metric_values: dict = {}
                    for x_batch_val in validation_data:
                        val_metric_values = self.the_model.test_step(x_batch_val)

                    l_test: float = val_metric_values["pref. loss"].numpy()
                    msg += f"\t\tError en test:\t{l_test:.6f}\n"

                    # Reset testing metrics at the end of validation
                    for m in self.the_model.metrics:
                        m.reset_states()

                    # Ahora hacemos las predicciones de pares de preferencias
                    # para el último usuario, que es el usuario interactivo. En este objeto
                    # ya contamos con el usuario interactivo, así que aquí su índice es self.num_users-1, mientras
                    # que en la clase de la aplicación su índice es num_users (allí num_users vale 1 menos que aquí)
                    # todo: Dos contadores para número de usuarios con valores distintos es confuso, hay que cambiarlo
                    l_iu: float = np.nan
                    iu_metric_values: dict = {}
                    if iu_data is not None:
                        # pero sólo si ha dado algunas puntuaciones
                        for interactive_user_data in iu_data:
                            iu_metric_values = self.the_model.test_step(
                                interactive_user_data
                            )

                        l_iu = iu_metric_values["pref. loss"].numpy()
                        msg += f"\t\tError en gustos del usuario: {l_iu:.6f}\n"
                        # Reset metrics at the end of testing
                        for m in self.the_model.metrics:
                            m.reset_states()

                    # emitimos señal con los datos que se pasan como parámetros y además, emite
                    # el usuario interactivo (última columna de W) y las películas (matriz V)
                    self._emite_datos(self.global_step, avg_loss, l_test, l_iu)
                    # self.average_loss = 0

                    t2 = time()
                    average_time = (t2 - t1) / steps_between_updates  # self.drawevery
                    estimated_time = average_time * (
                        (steps_per_epoch - step - 1)
                        + steps_per_epoch * (self.num_epochs - epoch - 1)
                    )
                    t1 = time()
                    msg += "(Tiempo restante estimado: %s)\n\n" % timedelta(
                        seconds=round(estimated_time)
                    )

                    steps_between_updates = 0

                    self.mensaje.emit(msg)

                    if self.parar_entrenamiento:
                        # self.parar_entrenamiento = False
                        self.mensaje.emit("\n *** ENTRENAMIENTO INTERRUMPIDO ***\n")
                        # self.autosave()
                        self.entrenamiento_finalizado.emit()
                        return  # salimos del entrenamiento

                porcentaje_completado = (
                    100
                    * (1 + step + steps_per_epoch * epoch)
                    / (self.num_epochs * steps_per_epoch)
                )
                self.progreso.emit(int(porcentaje_completado))

            # Reset training metrics at the end of each epoch
            # TODO: POSIBLEMENTE ESTO ES INNECESARIO, PUESTO QUE YA ESTÁ RESETEADO...

            for m in self.the_model.metrics:
                m.reset_states()

        t_medio /= contador
        print(f"TIEMPO MEDIO POR STEP: {t_medio} - STEPS: {contador}")

        self.mensaje.emit("\n--- ENTRENAMIENTO FINALIZADO ---\n")
        self.entrenamiento_finalizado.emit()
    """

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

        # self.the_model = tf.keras.models.load_model(path)
        # # Y ahora tenemos que rellenar correctamente los parámetros K y nu de Movielens_Learner,
        # # los demás forman parte del experimento en cada momento, no del modelo
        # if self.the_model is None:
        #     raise RuntimeError("Modelo no cargado por alguna razón")

        # laW: tf.keras.layers.Embedding = self.the_model.get_layer(name="W")
        # self.K = laW.output_dim
        # self.nu = laW.embeddings_regularizer.get_config()["l2"]
        # opt = self.the_model.optimizer
        # self.learning_rate = opt.get_config()["learning_rate"]

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

    def main():
        # Desabilitar GPU, en Mac M1/M2 no va fino :(
        tf.config.set_visible_devices([], "GPU")

        # Los conjuntos de entrenamiento y test van "atornillados" en el código, deben ser
        # pj_train.csv y pj_test.csv respectivamente
        train_pj, new_ucodes, new_mcodes = load_and_recode_pj("pj_train.csv")
        num_users = len(new_ucodes)
        num_movies = len(new_mcodes)
        # Los ejemplos de test se recodifican utilizando la codificación obtenida sobre los de entrenamiento, por
        # lo que es necesario que en el conjunto de entrenamiento aparezcan todos los usuarios y todas las películas
        # alguna vez, pero eso NO QUIERE DECIR que haya ejemplos de test en el conjunto de entrenamiento
        test_pj, _, _ = load_and_recode_pj("pj_test.csv", new_ucodes, new_mcodes)

        MV_learner: Movielens_Learner = Movielens_Learner(
            num_users + 1,
            num_movies,
            K=30,
            learning_rate=0.01,
            nu=0.000001,
            num_epochs=10,
            batch_size=25000,
        )

        # tf.keras.utils.plot_model(MV_learner.the_model, to_file="kk.png")
        MV_learner.fit(train_pj, test_pj)
        print(f"W=\n{MV_learner.W_weights}")
        print(f"V=\n{MV_learner.V_weights}")

        # cambio nu, entreno otro poco y los pesos deberían ser mucho más pequeños
        MV_learner.set_params(nu=0.1)
        MV_learner.fit(train_pj, test_pj)
        print(f"W=\n{MV_learner.W_weights}")
        print(f"V=\n{MV_learner.V_weights}")

        MV_learner.save("./modelo_guardado/modelo")

        MV_l2: Movielens_Learner = Movielens_Learner(
            num_users + 1,
            num_movies
        )

        MV_l2.restore_model("./modelo_guardado/modelo")
        print(f"W=\n{MV_learner.W_weights}")
        print(f"V=\n{MV_learner.V_weights}")

        print("THE END")

        # a = tf.keras.layers.Input(shape=(3,))
        # b = tf.keras.layers.Dense(units=10)(a)
        # m = tf.keras.Model(inputs=a, outputs=b)
        # tf.keras.models.save_model(m, "./modelo_guardado")

    main()
