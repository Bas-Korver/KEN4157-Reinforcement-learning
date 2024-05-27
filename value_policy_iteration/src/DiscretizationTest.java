import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class DiscretizationTest {
    int[] posIndex = {0, 500, 999, 821, 700};
    double[] discretizedPos = new double[posIndex.length];

    @BeforeEach
    void discretize() {
//        for (int i = 0; i < posIndex.length; i++) {
//            discretizedPos[i] = Discretization.discretize(posIndex[i], 1000, 0);
//        }
        System.out.println(Discretization.discretize(548, 1000, 0));
    }

    @Test
    void findDiscretization() {
//        int[] pos = new int[discretizedPos.length];
//        for (int i = 0; i < discretizedPos.length; i++) {
//            pos[i] = Discretization.findIndices(discretizedPos[i], 1000, 0);
//        }
//        assert Arrays.equals(pos, posIndex);
        System.out.println(Discretization.findIndices(-0.21261261201261266, 1000, 0));
    }
}