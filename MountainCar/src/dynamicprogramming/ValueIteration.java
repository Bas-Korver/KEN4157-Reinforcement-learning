package dynamicprogramming;

import environment.MountainCarEnv;

public class ValueIteration {
    public static final int POS_DISCRETIZATION_FACTOR = 1000;
    public static final int SPEED_DISCRETIZATION_FACTOR = 1000;
    private static final int[] ACTIONS_SET = {MountainCarEnv.REVERSE, MountainCarEnv.NOTHING, MountainCarEnv.FORWARD};
    private static final double THETA = 1e-15;
    private static final double GAMMA = 0.99;

    public static void run() {
        MountainCarEnv game = new MountainCarEnv(0);
        State.initializeValues(POS_DISCRETIZATION_FACTOR, SPEED_DISCRETIZATION_FACTOR);
        State.initializePolicy(POS_DISCRETIZATION_FACTOR, SPEED_DISCRETIZATION_FACTOR);

        int iterations = 0;
        double delta;
        do {
            double[][] valuesNew = new double[POS_DISCRETIZATION_FACTOR][SPEED_DISCRETIZATION_FACTOR];
            int[][] policyNew = new int[POS_DISCRETIZATION_FACTOR][SPEED_DISCRETIZATION_FACTOR];
            delta = 0;

            for (int i = 0; i < POS_DISCRETIZATION_FACTOR; i++) {
                double position = Discretization.discretize(i, POS_DISCRETIZATION_FACTOR, Discretization.CALCULATE_POS);

                for (int j = 0; j < SPEED_DISCRETIZATION_FACTOR; j++) {
                    double velocity = Discretization.discretize(j, SPEED_DISCRETIZATION_FACTOR, Discretization.CALCULATE_SPEED);

//                    double value = State.getValue(i, j);
                    double bestValue = Integer.MIN_VALUE;
                    int bestAction = Integer.MIN_VALUE;

                    for (int action : ACTIONS_SET) {
                        double[] state = game.setState(position, velocity);
                        if (state[0] != 1) {
                            double[] gameState = game.step(action);
                            int posIndex = Discretization.findDiscretization(gameState[2], POS_DISCRETIZATION_FACTOR, Discretization.CALCULATE_POS);
                            int velocityIndex = Discretization.findDiscretization(gameState[3], SPEED_DISCRETIZATION_FACTOR, Discretization.CALCULATE_SPEED);
                            double temp = gameState[1] + GAMMA * State.getValue(posIndex, velocityIndex);
                            if (temp > bestValue) {
                                bestValue = temp;
                                bestAction = action;
                            }
                        }
                    }
//                    State.setValue(i, j, bestValue);
//                    State.setPolicy(i, j, bestAction);
                    valuesNew[i][j] = bestValue;
                    policyNew[i][j] = bestAction;
                    delta = Math.max(delta, Math.abs(State.getValue(i, j) - bestValue));
                }
            }
            System.out.println("Iteration: " + iterations + "| Error: " + delta);
            iterations++;
            State.values = valuesNew;
            State.policy = policyNew;
        } while (delta > THETA);
    }

}
