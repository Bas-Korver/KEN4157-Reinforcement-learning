import javax.swing.*;
import java.util.Arrays;

public class DemoAgentVI {

    private static MountainCarEnv game;
    private static double[] gamestate;

    public static void main(String[] args) throws Exception {
        ValueIteration.run();
        game = new MountainCarEnv(MountainCarEnv.RENDER);

        //Running 5 episodes
        for (int i=0; i<5; i++) {
            gamestate = game.randomReset();
            System.out.println("The initial gamestate is: " + Arrays.toString(gamestate));
            long startTime = System.currentTimeMillis();

            while (gamestate[0] != 1.0 && System.currentTimeMillis() < startTime + 10000L) { // Game is not over yet
                int posIndex = Discretization.findIndices(gamestate[2], ValueIteration.POS_DISCRETIZATION_FACTOR, Discretization.CALCULATE_POS);
                int speedIndex = Discretization.findIndices(gamestate[3], ValueIteration.SPEED_DISCRETIZATION_FACTOR, Discretization.CALCULATE_SPEED);
                int action = State.getPolicy(posIndex, speedIndex);

                System.out.println("The car's position is " + gamestate[2]);
                System.out.println("The car's velocity is " + gamestate[3]);
                System.out.println("The indexes are " + posIndex + ", " + speedIndex);
                System.out.println("The car executes action: " + action);

                gamestate = game.step(action);

                System.out.println("The gamestate passed back to me was: " + Arrays.toString(gamestate));
                System.out.println("I received a reward of " + gamestate[1]);
            }
            System.out.println();
        }
//        try {
//            HeatMapWindow hm = new HeatMapWindow(State.getValues());
//            hm.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
//            hm.setSize(600,600);
//            hm.setVisible(true);
//            hm.update(State.getValues());
//        }
//        catch (Exception e) {System.out.println(e.getMessage());}
    }
}
