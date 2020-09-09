package main.env;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;

import javax.swing.JFrame;
import javax.swing.JPanel;

public abstract class EnvVisualizer extends JPanel {
    private static final long serialVersionUID = -1L;
    private final int pause_per_frame;

    public EnvVisualizer(String name, int screen_width, int screen_height, int fps) {
        this.pause_per_frame = 1000 / fps;
        this.setSize(screen_width, screen_height);

        JFrame frame = new JFrame(name);
        frame.setSize(screen_width, screen_height + 50);
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

        paint(g2d);
    }

    public void update(float[] state) {
        updateState(state);
        try {
            repaint();
            if (pause_per_frame > 1) {
                Thread.sleep(pause_per_frame);
            }
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
    }

    protected abstract void paint(Graphics2D g2d);

    protected abstract void updateState(float[] state);
}
