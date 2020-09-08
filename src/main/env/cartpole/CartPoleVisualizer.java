package main.env.cartpole;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.geom.Line2D;

import javax.swing.JFrame;
import javax.swing.JPanel;

public final class CartPoleVisualizer extends JPanel {
    private static final long serialVersionUID = -2201770607777176825L;
    private static final int SCREEN_WIDTH = 600;
    private static final int SCREEN_HEIGHT = 400;
    private static final int POLE_WIDTH = 10;
    private static final int CARTY = 300;
    private static final int CART_WIDTH = 50;
    private static final int CART_HEIGHT = 30;
    private static final double SCALE = SCREEN_HEIGHT;

    private final int pole_len;
    private final double x_threshold;
    private final int pause_per_frame;
    private float cart_location = 0.0F;
    private float pole_theta = (float) (Math.PI * 0.5);

    public CartPoleVisualizer(double pole_len, double x_threshold, int fps) {
        this.pole_len = (int) (SCALE * 0.5 * pole_len);
        this.x_threshold = x_threshold;
        this.pause_per_frame = 1000 / fps;
        this.setSize(SCREEN_WIDTH, SCREEN_HEIGHT);

        JFrame frame = new JFrame("CartPole");
        frame.setSize(SCREEN_WIDTH, SCREEN_HEIGHT + 50);
        frame.add(this);
        frame.setLocationRelativeTo(null);
        frame.getContentPane().setBackground(Color.WHITE);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setVisible(true);
    }

    @Override
    protected void paintComponent(Graphics graphics) {
        super.paintComponent(graphics);
        Graphics2D g2d = (Graphics2D) graphics.create();
        g2d.clearRect(0, 0, getWidth(), getHeight());
        g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

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

    public void update(float location, float theta) {
        try {
            this.cart_location = location;
            this.pole_theta = (float) (theta + 0.5 * Math.PI);
            repaint();
            if (pause_per_frame > 1) {
                Thread.sleep(pause_per_frame);
            }
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }

    }

}
