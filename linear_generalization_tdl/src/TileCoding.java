public class TileCoding {
    private static final double[] POS_RANGE = {MountainCarEnv.MIN_POS, MountainCarEnv.MAX_POS};
    private static final double[] SPEED_RANGE = {-MountainCarEnv.MAX_SPEED, MountainCarEnv.MAX_SPEED};
    public static int[] tileCoding(double position, double velocity, int numTilings, int numPosTiles, int numVelTiles) {
        int[] tiles = new int[numPosTiles * numVelTiles * numTilings];

        double tileWidth = (POS_RANGE[1] - POS_RANGE[0]) / numPosTiles;
        double tileHeight = (SPEED_RANGE[1] - SPEED_RANGE[0]) / numVelTiles;
        double posOffset = tileWidth / numTilings;
        double velOffset = tileHeight / numTilings;

        for (int tiling = 0; tiling < numTilings; tiling++) {
            int posIndex = (int)((position - POS_RANGE[0] - tiling * posOffset) / tileWidth);
            int velIndex = (int)((velocity - SPEED_RANGE[0] - tiling * velOffset) / tileHeight);

            // Edge case handling.
            if (posIndex < 0) posIndex = 0;
            if (posIndex >= numPosTiles) posIndex = numPosTiles - 1;
            if (velIndex < 0) velIndex = 0;
            if (velIndex >= numVelTiles) velIndex = numVelTiles - 1;

            int tile = tiling * (numPosTiles * numVelTiles) + velIndex * numPosTiles + posIndex;
            tiles[tile] = 1;
        }

        return tiles;
    }
}




