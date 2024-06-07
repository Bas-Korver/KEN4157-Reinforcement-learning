import javax.swing.*;
import java.util.Arrays;
import java.util.Random;

public class QLearning {
    public static final int POS_TILES = 10;
    public static final int SPEED_TILES = 10;
    public static final int TILING_AMOUNT = 7;
    public static final int[] ACTIONS_SET = {MountainCarEnv.REVERSE, MountainCarEnv.NOTHING, MountainCarEnv.FORWARD};
    private static final double EPSILON = 1e-1;
    private static final double ALPHA = 0.1;
    private static final double GAMMA = 0.99;
    private static final double LAMBDA = 0.3;
    private static final int MAX_EPISODES = 1000;

    public static int[][] createFeatures(int[] tiles) {
        int[][] features = new int[ACTIONS_SET.length][tiles.length * ACTIONS_SET.length + 1];
        for (int[] row : features) {
            Arrays.fill(row, 0);
        }

        for (int i = 0; i < ACTIONS_SET.length; i++) {
            System.arraycopy(tiles, 0, features[i], i * tiles.length, tiles.length);
            features[i][features[i].length - 1] = 1;
        }

        return features;
    }

    private static int epsilonGreedy(double[] state, double[] weights) {
        if (Math.random() < EPSILON) {
            return new Random().nextInt(ACTIONS_SET.length);
        } else {
            int maxIndex = 0;
            int[] tiles = TileCoding.tileCoding(state[2], state[3], TILING_AMOUNT, POS_TILES, SPEED_TILES);
            int[][] features = createFeatures(tiles);
            double[] values = new double[ACTIONS_SET.length];

            for (int x = 0; x < ACTIONS_SET.length; x++) {
                for (int j = 0; j < weights.length; j++) {
                    values[x] += weights[j] * features[x][j];
                }
            }

            for (int x = 0; x < values.length; x++) {
                maxIndex = values[x] > values[maxIndex] ? x : maxIndex;
            }
            return maxIndex;
        }
    }

    private static double calculateValue(double[] weights, int[] features) {
        double value = 0;
        for (int i = 0; i < weights.length; i++) {
            value += weights[i] * features[i];
        }
        return value;
    }

    public static double[] run() throws Exception {
        MountainCarEnv game = new MountainCarEnv(0);
        double[] weights = new double[ACTIONS_SET.length * TILING_AMOUNT * POS_TILES * SPEED_TILES + 1];

        for (int i = 0; i < MAX_EPISODES; i++) {
            double[] state = game.randomReset();
            boolean terminal = false;
            int iterations = 0;
            while (!terminal && iterations < 1000) {
                int action = epsilonGreedy(state, weights);
                double[] nextState = game.step(ACTIONS_SET[action]);
                terminal = nextState[0] > 0;

                int[] tiles = TileCoding.tileCoding(state[2], state[3], TILING_AMOUNT, POS_TILES, SPEED_TILES);
                int[] nextTiles = TileCoding.tileCoding(nextState[2], nextState[3], TILING_AMOUNT, POS_TILES, SPEED_TILES);
                int[][] features = createFeatures(tiles);
                int[][] nextFeatures = createFeatures(nextTiles);

                double currentValue = calculateValue(weights, features[action]);
                double nextValue = terminal ? 0 : calculateValue(weights, nextFeatures[action]);
                double delta = nextState[1] + GAMMA * nextValue - currentValue;

                for (int j = 0; j < weights.length; j++) {
                    if (features[action][j] != 0) {
                        weights[j] += ALPHA * delta;
                    }
                }

                // Prepare for next step
                state = nextState;
                iterations++;
            }
            System.out.println("Episode " + i + " completed.");

        }
        // Initialise heatmap
        State.initializeValues(1000, 1000);
        HeatMapWindow hm = new HeatMapWindow(State.getValues());
        hm.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        hm.setSize(600,600);
        hm.setVisible(true);

        CalculateHeatmap(weights, hm);

        return weights;
    }

    private static void CalculateHeatmap(double[] weights, HeatMapWindow hm) {
        // Get values for discretized states
        for (int i = 0; i < 1000; i++) {
            double position = Discretization.discretize(i, 1000, Discretization.CALCULATE_POS);
            for (int j = 0; j < 1000; j++) {
                double velocity = Discretization.discretize(j, 1000, Discretization.CALCULATE_SPEED);
                int[] tiles = TileCoding.tileCoding(position, velocity, TILING_AMOUNT, POS_TILES, SPEED_TILES);
                int[][] features = createFeatures(tiles);
                int maxIndex = 0;
                double[] values = new double[ACTIONS_SET.length];

                for (int x = 0; x < ACTIONS_SET.length; x++) {
                    for (int y = 0; y < weights.length; y++) {
                        values[x] += weights[y] * features[x][y];
                    }
                }

                for (int x = 0; x < values.length; x++) {
                    maxIndex = values[x] > values[maxIndex] ? x : maxIndex;
                }

                State.setValue(i, j, calculateValue(weights, features[maxIndex]));

            }
        }
        hm.update(State.getValues());
        System.out.println("Done");


    }

}
