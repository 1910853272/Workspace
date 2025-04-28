<<<<<<< HEAD
package com.itheima.d7_tcp4;

import java.io.DataInputStream;
import java.io.InputStream;
import java.net.Socket;

public class ClentReaderThread extends Thread{
    private Socket socket;
    public ClentReaderThread(Socket socket){
        this.socket = socket;
    }
    @Override
    public void run() {
        try {
            InputStream is = socket.getInputStream();
            DataInputStream dis = new DataInputStream(is);
            while (true){
                try {
                    String msg = dis.readUTF();
                    System.out.println(msg);
                } catch (Exception e) {
                    System.out.println("自己下线了：" + socket.getRemoteSocketAddress());
                    dis.close();
                    socket.close();
                    break;
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
=======
package com.itheima.d7_tcp4;

import java.io.DataInputStream;
import java.io.InputStream;
import java.net.Socket;

public class ClentReaderThread extends Thread{
    private Socket socket;
    public ClentReaderThread(Socket socket){
        this.socket = socket;
    }
    @Override
    public void run() {
        try {
            InputStream is = socket.getInputStream();
            DataInputStream dis = new DataInputStream(is);
            while (true){
                try {
                    String msg = dis.readUTF();
                    System.out.println(msg);
                } catch (Exception e) {
                    System.out.println("自己下线了：" + socket.getRemoteSocketAddress());
                    dis.close();
                    socket.close();
                    break;
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
>>>>>>> d418ed5a0517a1c34a75d286fd5c685a031f6eb6
