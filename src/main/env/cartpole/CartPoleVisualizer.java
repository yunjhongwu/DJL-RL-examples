package main.env.cartpole;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.geom.Line2D;

import main.env.BaseVisualizer;

public final class CartPoleVisualizer extends BaseVisualizer {
    private static final long serialVersionUID = -1L;
    private static final int SCREEN_WIDTH = 600;
    private static final int SCREEN_HEIGHT = 400;
    private static final int POLE_WIDTH = 10;
    private static final int CARTY = 300;
    private static final int CART_WIDTH = 50;
    private static final int CART_HEIGHT = 30;
    private static final double SCALE = SCREEN_HEIGHT;

    private final int pole_len;
    private final double x_threshold;
    private float cart_location = 0.0F;
    private float pole_theta = (float) (Math.PI * 0.5);

    public CartPoleVisualizer(double pole_len, double x_threshold, int fps) {
        super("CartPole", SCREEN_WIDTH, SCREEN_HEIGHT, fps);
        this.pole_len = (int) (SCALE * 0.5 * pole_len);
        this.x_threshold = x_threshold;
    }

    @Override
    protected void paint(Graphics2D g2d) {
        int location = (int) (SCREEN_WIDTH * (cart_location / (2 * x_threshold) + 0.5));
        int cart_left = (int) (-CART_WIDTH * 0.5) + location;
        int cart_top = (int) (-CART_HEIGHT * 0.5) + CARTY;

        g2d.setColor(Color.LIGHT_GRAY);
        g2d.fillRect(0, CARTY, SCREEN_WIDTH, 2);
        g2d.setColor(Color.BLACK);
        g2d.fillRect(cart_left, cart_top, CART_WIDTH, CART_HEIGHT);
        g2d.setColor(Color.ORANGE);
        g2d.setStroke(new BasicStroke(POLE_WIDTH));
        g2d.draw(new Line2D.Double(location, CARTY, location + pole_len * Math.cos(pole_theta),
                CARTY - pole_len * Math.sin(pole_theta)));
        g2d.setColor(Color.MAGENTA);
        g2d.fillOval(location - 5, CARTY - 5, 10, 10);
        g2d.dispose();
    }

    @Override
    protected void updateState(float[] state) {
        cart_location = state[0];
        pole_theta = (float) (state[2] + 0.5 * Math.PI);
    }

}
