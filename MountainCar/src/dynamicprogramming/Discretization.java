package dynamicprogramming;

import environment.MountainCarEnv;

public class Discretization {
    public static final int CALCULATE_POS = 0;
    public static final int CALCULATE_SPEED = 1;
    private static final double[] POS_RANGE = {MountainCarEnv.MIN_POS, MountainCarEnv.MAX_POS};
    private static final double[] SPEED_RANGE = {-MountainCarEnv.MAX_SPEED, MountainCarEnv.MAX_SPEED};

    // Discretize state based on index, discretization step size and which range to discretize.
    public static double discretize(int index, int discretizationFactor, int rangeToDiscretize) {
        double discretizeStep;
        if (rangeToDiscretize == CALCULATE_POS) {
            discretizeStep = (POS_RANGE[1] - POS_RANGE[0]) / (discretizationFactor - 1);
            return POS_RANGE[0] + discretizeStep * index;

        } else if (rangeToDiscretize == CALCULATE_SPEED) {
            discretizeStep = (SPEED_RANGE[1] - SPEED_RANGE[0]) / (discretizationFactor - 1);
            return SPEED_RANGE[0] + discretizeStep * index;

        } else {
            System.out.println("Invalid discretization range");
            return 0;
        }
    }

    public static int findDiscretization(double state, int discretizationFactor, int rangeToDiscretize) {
        double discretizeStep;
        if (rangeToDiscretize == CALCULATE_POS) {
            discretizeStep = (POS_RANGE[1] - POS_RANGE[0]) / (discretizationFactor - 1);
            return (int) Math.round((state - POS_RANGE[0]) / discretizeStep);

        } else if (rangeToDiscretize == CALCULATE_SPEED) {
            discretizeStep = (SPEED_RANGE[1] - SPEED_RANGE[0]) / (discretizationFactor - 1);
            return (int) Math.round((state - SPEED_RANGE[0]) / discretizeStep);

        } else {
            System.out.println("Invalid discretization range");
            return 0;
        }
    }


    // Inefficient
    // Use binary search to find the nearest discrete state sources used:
    // https://www.geeksforgeeks.org/java-program-to-find-closest-number-in-array/
    // and https://en.wikipedia.org/wiki/Binary_search_algorithm
//    private static int binarySearch(double key, int low, int high, int discretizationFactor, int rangeToDiscretize) {
//        int mid = (low + high) >> 1;
//        double midValue = discretize(mid, discretizationFactor, rangeToDiscretize);
//
//        if (key < midValue) {
//            double valueLeftMid = discretize(mid - 1, discretizationFactor, rangeToDiscretize);
//
//            if (key > valueLeftMid) {
//                if (findNearest(valueLeftMid, midValue, key)) return mid - 1;
//                else return mid;
//            }
//
//            high = mid - 1;
//            return binarySearch(key, low, high, discretizationFactor, rangeToDiscretize);
//
//        } else if (key > midValue) {
//            double valueRightMid = discretize(mid + 1, discretizationFactor, rangeToDiscretize);
//
//            if (key < valueRightMid) {
//                if (findNearest(valueRightMid, midValue, key)) return mid + 1;
//                else return mid;
//            }
//
//            low = mid + 1;
//            return binarySearch(key, low, high, discretizationFactor, rangeToDiscretize);
//        } else {
//            return mid;
//        }
//    }

//    public static int binarySearch(double key, int discretizationFactor, int rangeToDiscretize) {
//        return binarySearch(key, 0, discretizationFactor, discretizationFactor, rangeToDiscretize);
//    }

    // Calculate which of the two values is nearest to the given state(key)
//    private static boolean findNearest(double firstValue, double secondValue, double key) {
//        return Math.abs(firstValue - key) <= Math.abs(secondValue - key);
//    }
}
