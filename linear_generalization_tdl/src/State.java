import java.util.Random;

// Utility class
public class State {
    private static double[][] values;
    private static int[][] policy;

    public static int getPolicy(int row, int column) {
        return policy[row][column];
    }

    public static void setPolicy(int row, int column, int policies) {
        policy[row][column] = policies;
    }

    public static double getValue(int row, int column) {
        return values[row][column];
    }

    public static void setValue(int row, int column, double value) {
        values[row][column] = value;
    }

    public static double[][] getValues() {
        return values;
    }

    public static void initializeValues(int rows, int columns) {
        values = new double[rows][columns];
//        Random r = new Random();
//        int min = -99;

        for (int i = 0; i < rows; i++){
            for (int j = 0; j < columns; j++) {
                // Random initialization
//                double discretizedPos = Discretization.discretize(i, rows, Discretization.CALCULATE_POS);
//                if (discretizedPos > MountainCarEnv.GOAL_POS) {
//                    values[i][j] = 0.0;
//                }
//                else {
//                    int minimalMovesToGoal = (int) Math.round((MountainCarEnv.GOAL_POS - discretizedPos) / MountainCarEnv.MAX_SPEED);
//                    double max = minimalMovesToGoal * -1.0;
//                    values[i][j] = r.nextDouble(max - min) + min;
//                }
                values[i][j] = 0.0;
            }
        }
    }

    public static void initializePolicy(int rows, int columns) {
        policy = new int[rows][columns];
    }
}
