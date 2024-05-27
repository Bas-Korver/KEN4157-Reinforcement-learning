public class ValueIteration {
    public static final int POS_DISCRETIZATION_FACTOR = 1000;
    public static final int SPEED_DISCRETIZATION_FACTOR = 1000;
    private static final int[] ACTIONS_SET = {MountainCarEnv.FORWARD, MountainCarEnv.REVERSE, MountainCarEnv.NOTHING};
    private static final double THETA = 1e-20;
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
                    double[] state = game.setState(position, velocity);

                    if (state[0] == 1) {
                        continue;
                    }

//                    double value = State.getValue(i, j);
                    double bestValue;
                    double[] actionValues = new double[ACTIONS_SET.length];
                    int bestAction;


                    for (int a = 0; a < ACTIONS_SET.length; a++) {
                        double[] gameState = game.step(ACTIONS_SET[a]);
                        int posIndex = Discretization.findIndices(gameState[2], POS_DISCRETIZATION_FACTOR, Discretization.CALCULATE_POS);
                        int velocityIndex = Discretization.findIndices(gameState[3], SPEED_DISCRETIZATION_FACTOR, Discretization.CALCULATE_SPEED);
//                        actionValues[a] = gameState[1] + GAMMA * State.getValue(posIndex, velocityIndex);
                        actionValues[a] = gameState[1] + GAMMA * State.values[posIndex][velocityIndex];
                    }

                    int maxIndex = 0;
                    for (int x = 0; x < actionValues.length; x++) {
                        maxIndex = actionValues[x] > actionValues[maxIndex] ? x : maxIndex;
                    }
                    bestValue = actionValues[maxIndex];
                    bestAction = ACTIONS_SET[maxIndex];


//                    State.setValue(i, j, bestValue);
//                    State.setPolicy(i, j, bestAction);
//                    delta = Math.max(delta, Math.abs(value - bestValue));
                    valuesNew[i][j] = bestValue;
                    policyNew[i][j] = bestAction;
//                    delta = Math.max(delta, Math.abs(State.getValue(i, j) - bestValue));
                    delta = Math.max(delta, Math.abs(State.values[i][j] - bestValue));
                }
            }
            System.out.println("Iteration: " + iterations + "| Error: " + delta);
            iterations++;
            State.values = valuesNew;
            State.policy = policyNew;
        } while (delta > THETA);

        double[][] temp1 = State.values;
        int[][] temp2 = State.policy;
        System.out.println();
    }

}
