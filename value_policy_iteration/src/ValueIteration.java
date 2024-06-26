import javax.swing.*;

public class ValueIteration {
    public static final int POS_DISCRETIZATION_FACTOR = 1000;
    public static final int SPEED_DISCRETIZATION_FACTOR = 1000;
    private static final int[] ACTIONS_SET = {MountainCarEnv.REVERSE, MountainCarEnv.NOTHING, MountainCarEnv.FORWARD};
    private static final double THETA = 1e-20;
    private static final double GAMMA = 0.99;

    public static void run() throws Exception {
        MountainCarEnv game = new MountainCarEnv(0);
        State.initializeValues(POS_DISCRETIZATION_FACTOR, SPEED_DISCRETIZATION_FACTOR);
        State.initializePolicy(POS_DISCRETIZATION_FACTOR, SPEED_DISCRETIZATION_FACTOR);

        // Initialise heatmap
        HeatMapWindow hm = new HeatMapWindow(State.getValues());
        hm.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        hm.setSize(600,600);
        hm.setVisible(true);

        int iterations = 0;
        double delta;
        do {
            delta = 0;
            for (int i = 0; i < POS_DISCRETIZATION_FACTOR; i++) {
                double position = Discretization.discretize(i, POS_DISCRETIZATION_FACTOR, Discretization.CALCULATE_POS);

                for (int j = 0; j < SPEED_DISCRETIZATION_FACTOR; j++) {
                    double velocity = Discretization.discretize(j, SPEED_DISCRETIZATION_FACTOR, Discretization.CALCULATE_SPEED);
                    double[] state = game.setState(position, velocity);

                    if (state[0] == 1) {
                        continue;
                    }

                    double value = State.getValue(i, j);
                    double bestValue;
                    double[] actionValues = new double[ACTIONS_SET.length];
                    int bestAction;


                    for (int a = 0; a < ACTIONS_SET.length; a++) {
                        double[] gameState = game.step(ACTIONS_SET[a]);
                        int posIndex = Discretization.findIndices(gameState[2], POS_DISCRETIZATION_FACTOR, Discretization.CALCULATE_POS);
                        int velocityIndex = Discretization.findIndices(gameState[3], SPEED_DISCRETIZATION_FACTOR, Discretization.CALCULATE_SPEED);
                        actionValues[a] = gameState[1] + GAMMA * State.getValue(posIndex, velocityIndex);
                        game.setState(position, velocity);
                    }

                    int maxIndex = 0;
                    for (int x = 0; x < actionValues.length; x++) {
                        maxIndex = actionValues[x] > actionValues[maxIndex] ? x : maxIndex;
                    }
                    bestValue = actionValues[maxIndex];
                    bestAction = ACTIONS_SET[maxIndex];


                    State.setValue(i, j, bestValue);
                    State.setPolicy(i, j, bestAction);
                    delta = Math.max(delta, Math.abs(value - bestValue));
                }
            }

            hm.update(State.getValues());

            System.out.println("Iteration: " + iterations + "| Error: " + delta);
            iterations++;
        } while (delta > THETA);

        System.out.println();
    }

}
