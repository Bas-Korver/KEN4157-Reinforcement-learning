import java.util.Arrays;

public class DemoAgentLG {

    private static MountainCarEnv game;
    private static double[] gamestate;

    public static void main(String[] args) throws Exception {
        double[] weights = QLearning.run();
        game = new MountainCarEnv(MountainCarEnv.RENDER);

        //Running 5 episodes
        for (int i=0; i<5; i++) {
            gamestate = game.randomReset();
            System.out.println("The initial gamestate is: " + Arrays.toString(gamestate));
            long startTime = System.currentTimeMillis();

//             && System.currentTimeMillis() < startTime + 20000L
            while (gamestate[0] != 1.0) { // Game is not over yet
                int maxIndex = 0;
                int[] tiles = TileCoding.tileCoding(gamestate[2], gamestate[3], SarsaLambda.TILING_AMOUNT, SarsaLambda.POS_TILES, SarsaLambda.SPEED_TILES);
                int[][] features = SarsaLambda.createFeatures(tiles);
                double[] values = new double[SarsaLambda.ACTIONS_SET.length];

                for (int x = 0; x < SarsaLambda.ACTIONS_SET.length; x++) {
                    for (int j = 0; j < weights.length; j++) {
                        values[x] += weights[j] * features[x][j];
                    }
                }

                for (int x = 0; x < values.length; x++) {
                    maxIndex = values[x] > values[maxIndex] ? x : maxIndex;
                }

                System.out.println("The car's position is " + gamestate[2]);
                System.out.println("The car's velocity is " + gamestate[3]);
                System.out.println("The car executes action: " + SarsaLambda.ACTIONS_SET[maxIndex]);

                gamestate = game.step(SarsaLambda.ACTIONS_SET[maxIndex]);

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
