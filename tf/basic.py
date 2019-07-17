from base import ExperimentBase
import tensorflow as tf


class BasicTensorFlowExperiment(ExperimentBase):
    async def do_run_async(self):
        default_graph = tf.get_default_graph()
        # x and y variables added to the default graph
        x = tf.Variable(3, name="x")
        y = tf.Variable(5, name="y")

        print(f"x variable is part of the default graph: {x.graph == default_graph}")
        print(f"y variable is part of the default graph: {y.graph == default_graph}")

        # Try add f function and a new variable to a separate graph
        graph = tf.Graph()

        with graph.as_default():
            f = x * x * y + y + 2
            z = tf.Variable(3, name="z")

        print(f"f function is part of the default graph: {f.graph == default_graph}")
        print(f"f function is part of the new graph: {f.graph == graph}")

        print(f"z variable is part of the default graph: {z.graph == default_graph}")
        print(f"z variable is part of the new graph: {z.graph == graph}")

        # Only the new variable node, z will be added to the new graph since it's independent.
        # Even if defined in the scope of the new graph set as default,
        # the function will be added to the same graph as x and y variable nodes as it depends on them

        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            init.run()
            result = f.eval()
            print(result)

        x = tf.placeholder(tf.int32, shape=(None, 3))  # x is set with bi-dimensional shape but first size of the shape is not specified - could be anything
        y = tf.Variable(7, name="y")
        z = tf.placeholder(tf.int32, shape=(3,))
        f = x + y + z + 1

        with tf.Session() as sess:
            y.initializer.run()
            result = sess.run(f, feed_dict={x: [[1, 2, 3]], z: [1, 1, 1]})  # x of shape (1,3)
            print(f"Result for x of shape(1,3): {result}")  # result of shape (1,3)

            result = sess.run(f, feed_dict={x: [[1, 2, 3], [4, 5, 6]], z: [1, 1, 1]})  # x of shape (2,3)
            print(f"Result for x of shape(2,3): {result}") # result of shape (2,3)

        # Save a graph
        graph = tf.Graph()

        with graph.as_default():
            y = tf.Variable(5, name="y")
            saver = tf.train.Saver()

            with tf.Session() as sess:
                y.initializer.run()
                saver.save(sess, "BasicTensorFlowExperiment.dat")

        # Restore to a new graph
        graph2 = tf.Graph()

        with graph2.as_default():
            y = tf.Variable(0, name="y")
            saver = tf.train.Saver()

            with tf.Session() as sess:
                saver.restore(sess, "BasicTensorFlowExperiment.dat")
                f = y + 1  # y doesn't need to be initialized anymore as it's being restored with value 5 (from previous initialization) from disk persistence
                print(f.eval())
