/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package game;

/**
 *
 * @author lorenzosic
 */
public class Missile extends Sprite {

    private final int BOARD_WIDTH = 1000;
    private final int MISSILE_SPEED = 2;

    public Missile(int x, int y) {
        super(x, y);
        
        initMissile();
    }
    
    private void initMissile() {
        
        loadImage("src/resources/missile.png");  
        getImageDimensions();
    }

    public void move() {
        
        y -= MISSILE_SPEED;
        
        if (y > BOARD_WIDTH) {
            visible = false;
        }
    }
}
