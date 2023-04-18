from operator import le
from re import L
import numpy as np
import tensorflow as tf

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot
import pandas as pd
from datetime import timedelta
from time import time
import math
from copy import deepcopy

# class Movielens_PrefLoss(tf.keras.losses.Loss):
#     def call(self, best, worst):
#         return tf.math.maximum(0.0, 1.0 - best + worst)


class Movielens_Model(tf.keras.Model):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.loss_tracker = tf.keras.metrics.Mean(name="pref. loss")

    def train_step(self, data):
        u, b, w = data[:, 0:1], data[:, 1:2], data[:, 2:3]

        with tf.GradientTape() as tape:
            y_best, y_worst = self([u, b, w], training=True) # type: ignore
            # loss = self.compiled_loss(y_best, y_worst)
            loss = tf.math.maximum(0.0, 1.0 - y_best + y_worst)

        # Calcular gradientes
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Actualizar los pesos
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Calcular nuestra propia métrica
        self.loss_tracker.update_state(loss)

        # Retornar un dict mapeando los nombres de las métricas con su valores
        return {m.name: m.result() for m in self.metrics}

    @property
    def metrics(self):
        return [self.loss_tracker]

    def test_step(self, data):
        u, b, w = data[:, 0:1], data[:, 1:2], data[:, 2:3]
        y_best, y_worst = self([u, b, w], training=False) # type: ignore
        loss = tf.math.maximum(0.0, 1.0 - y_best + y_worst)
        self.loss_tracker.update_state(loss)

        # Retornar un dict mapeando los nombres de las métricas con su valores
        return {m.name: m.result() for m in self.metrics}


class Movielens_Learner(QObject):
    # Señal que se emite cada vez que sacamos información durante el entrenamiento
    computed_avg_loss = pyqtSignal(int, float, float, float)
    computed_embeddings = pyqtSignal(np.ndarray, np.ndarray)
    grafo_construido = pyqtSignal()
    grafo_eliminado = pyqtSignal()
    entrenamiento_finalizado = pyqtSignal()
    mensaje = pyqtSignal(str)
    progreso = pyqtSignal(int)

    def __init__(
        self,
        num_users,
        num_movies,
        K=100,
        learning_rate=0.01,
        nu=0.000001,
        num_epochs=2,
        batch_size=25,
        # batch_gen="random",
        drawevery=2,
        random_seed=True,
        # optimizer="Adam",
        # save_path="movielens_model",
    ) -> None:
        super().__init__()

        self.random_seed = random_seed
        self.learning_rate = learning_rate
        self.nu = nu
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        # self.batch_gen = batch_gen
        self.num_users = num_users
        self.num_movies = num_movies
        self.K = K
        # self.optimizer = optimizer
        # self.save_path = save_path

        # flag que se usa para interrumpir un entrenamiento
        self.parar_entrenamiento = False
        # # Lista que contiene los steps y errores medios
        # self.global_step = self.average_loss = 0
        self.global_step = 0
        self.drawevery = drawevery  # cada 'drawevery' iteraciones se dibuja

        # # La creación del grafo se deja para la primera vez que se intente entrenar el sistema
        self.the_model: Movielens_Model | None = None
        
        # Desabilitar GPU, en Mac M1/M2 no va fino :(
        # tf.config.set_visible_devices([], "GPU")

    def set_params(self, **params) -> None:
        # K no va a poder ser modificada, sólo se permite desde la interfaz gráfica
        # cuando aún no se ha creado el modelo, así que sólo me tengo que preocupar
        # de propagar hacia el modelo existente los cambios en learning_rate y nu
        for k, v in params.items():
            setattr(self, k, v)
            if self.the_model is not None:
                if k == "learning_rate":
                    config: dict = self.the_model.optimizer.get_config()
                    config["learning_rate"] = v
                    self.the_model.optimizer = self.the_model.optimizer.from_config(config)
                    # self.the_model.optimizer.lr.assign(v)
                elif k == "nu":
                    # TODO: esto también se podría hacer con get_config() / from_config()
                    for emb_name in ("W", "V"):
                        emb: tf.keras.layers.Embedding = self.the_model.get_layer(
                            name=emb_name
                        )
                        emb.embeddings_regularizer = tf.keras.regularizers.l2(v)


    def get_params(self) -> dict:
        hyperparams_list = [
            "K",
            "nu",
            "learning_rate",
            "batch_size",
            "num_epochs",
            "drawevery",
            "random_seed",
        ]

        params_actuales = {}
        for pname in hyperparams_list:
            params_actuales[pname] = getattr(self, pname)

        return params_actuales

    @property
    def W_embedding(self) -> tf.keras.layers.Embedding:
        return self.the_model.get_layer(name="W")
    
    @property
    def V_embedding(self) -> tf.keras.layers.Embedding:
        return self.the_model.get_layer(name="V")
    

    
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
        emb_reg = tf.keras.regularizers.L2(l2=self.nu)

        W = tf.keras.layers.Embedding(
            self.num_users,
            self.K,
            name="W",
            embeddings_initializer="uniform",
            embeddings_regularizer=emb_reg,
        )
        V = tf.keras.layers.Embedding(
            self.num_movies,
            self.K,
            name="V",
            embeddings_initializer="uniform",
            embeddings_regularizer=emb_reg,
        )

        Wu = W(input_user)
        Vb = V(best_movie)
        Vw = V(worst_movie)

        f_best = tf.keras.layers.Dot(axes=2, normalize=False, name="WuVb")([Wu, Vb])
        f_worst = tf.keras.layers.Dot(axes=2, normalize=False, name="WuVw")([Wu, Vw])
        outputs = [tf.linalg.diag_part(f_best), tf.linalg.diag_part(f_worst)]

        self.the_model = Movielens_Model(
            inputs=[input_user, best_movie, worst_movie], outputs=outputs
        )
        self.the_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            # loss = Movielens_PrefLoss # esto no lo doy echado a andar correctamente, la "empotro" en el fit()
        )
        self.global_step = 0
        # self.hay_grafo = True
        self.grafo_construido.emit()

    def reset_graph(self) -> None:
        self.the_model = None
        self.grafo_eliminado.emit()

    # def summary(self, *args, **kwargs):
    #     return self.the_model.summary(*args, **kwargs)

    # def fit(self, *args, **kwargs):
    #     return self.the_model.fit(*args, **kwargs)

    def _emite_datos(self, gs, avl, lt, lu):
        # emitimos los errores para el gráfico de errores
        self.computed_avg_loss.emit(gs, avl, lt, lu)
        # Emitimos los embeddings, para el gráfico de las películas
        W: np.ndarray
        movies: np.ndarray
        W, movies = self.getEmbeddings()
        # la ultima columna es el vector del usuario interactivo
        user: np.ndarray = W[:, -1]
        self.computed_embeddings.emit(user, movies)

    def getEmbeddings(self) -> tuple:
        return (
            self.W_embedding.get_weights()[0].transpose(),
            self.V_embedding.get_weights()[0].transpose(),
        )

    def stop_fit(self) -> None:
        self.parar_entrenamiento = True

    def fit(
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

        # TODO: ESTO DEBE ESTAR MAL !! ¿O NO? ¡¡COMPROBAR!!
        # También separamos los datos del usuario interactivo, si los hay, para ver
        # como se va comportando el modelo atendiendo únicamente a dicho usuario
        data_interactive_user = data[data.user == self.num_users - 1]
        print(f"self.num_users={self.num_users}")
        print(f"len(data_interactive_user)={len(data_interactive_user)}")

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
        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch}")
            # Iterar sobre los batches del conjunto de entrenamiento
            metric_values: dict = {}
            for step, x_batch_train in enumerate(training_data):
                # print(f"Training step {step}")
                # batch_size: int = x_batch_train.shape[0]
                metrics_values = self.the_model.train_step(x_batch_train)

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

        self.mensaje.emit("\n--- ENTRENAMIENTO FINALIZADO ---\n")
        self.entrenamiento_finalizado.emit()

    def predict(self, data, **kwargs):
        if self.the_model is None:
            raise RuntimeError("Modelo inexistente")

        predictions, _ = self.the_model.predict(
            [data.user, data.movie, data.movie], **kwargs
        )
        df_result = pd.DataFrame(columns=["user", "movie", "predicted_score"])
        df_result.user = data.user
        df_result.movie = data.movie
        df_result.predicted_score = predictions
        return df_result

    def restore_model(self, path) -> None:
        self.the_model = tf.keras.models.load_model(path, custom_objects={"Movielens_Model": Movielens_Model})
        # Y ahora tenemos que rellenar correctamente los parámetros K y nu de Movielens_Learner,
        # los demás forman parte del experimento en cada momento, no del modelo
        if self.the_model is None:
            raise RuntimeError("Modelo no cargado por alguna razón")
        
        laW : tf.keras.layers.Embedding = self.the_model.get_layer(name="W")
        self.K = laW.output_dim
        self.nu = laW.embeddings_regularizer.get_config()["l2"]
        opt = self.the_model.optimizer
        self.learning_rate = opt.get_config()["learning_rate"]
        # Y emitimos que ya hay modelo: se podrá grabar y exportar embeddings, 
        # además de seguir entrenando.
        self.grafo_construido.emit()



    def save(self, path:str)->None:
        if self.the_model is not None:
            self.the_model.save(path)


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

        # batch_size = 512
        # # Prepare the training dataset.
        # train_dataset = tf.data.Dataset.from_tensor_slices(train_pj)
        # train_dataset = train_dataset.shuffle(buffer_size=train_dataset.cardinality()).batch(batch_size)

        # # Prepare the validation dataset.
        # test_dataset = tf.data.Dataset.from_tensor_slices(test_pj)
        # test_dataset = test_dataset.batch(test_dataset.cardinality())

        MV_learner: Movielens_Learner = Movielens_Learner(
            num_users+1,  # hay que contar con el interactivo, para cuando lo haya
            num_movies,
            learning_rate=0.01,
            nu=1e-2,
            K=128,
            num_epochs=1,
            batch_size=2000,
        )

        MV_learner._init_graph()
        MV_learner.summary()

        tf.keras.utils.plot_model(MV_learner.the_model, to_file="kk.png")
        # print(list(train_dataset.as_numpy_iterator()))
        MV_learner.fit(data=train_pj, test_data=test_pj)
        MV_learner.save("./modelin")
        # MV_learner.fit(data=train_pj, test_data=test_pj)
        MV_learner.restore_model("./modelin")
        # tf.keras.utils.plot_model(MV_learner.the_model, to_file="kk-restored.png")
        MV_learner.fit(data=train_pj, test_data=test_pj)

    main()
