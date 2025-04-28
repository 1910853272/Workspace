<<<<<<< HEAD
package com.itheima.d8_tcp5;

import java.net.ServerSocket;
import java.net.Socket;

/**
 *  目标：完成TCP通信快速入门-服务端开发：要求实现与多个客户端同时通信。
 */
public class Server {
    public static void main(String[] args) throws Exception {
        System.out.println("-----服务端启动成功-------");
        // 1、创建ServerSocket的对象，同时为服务端注册端口。
        ServerSocket serverSocket = new ServerSocket(8080);

        while (true) {
            // 2、使用serverSocket对象，调用一个accept方法，等待客户端的连接请求
            Socket socket = serverSocket.accept();

            System.out.println("有人上线了：" + socket.getRemoteSocketAddress());

            // 3、把这个客户端对应的socket通信管道，交给一个独立的线程负责处理。
            new ServerReaderThread(socket).start();
        }
    }
}













=======
package com.itheima.d8_tcp5;

import java.net.ServerSocket;
import java.net.Socket;

/**
 *  目标：完成TCP通信快速入门-服务端开发：要求实现与多个客户端同时通信。
 */
public class Server {
    public static void main(String[] args) throws Exception {
        System.out.println("-----服务端启动成功-------");
        // 1、创建ServerSocket的对象，同时为服务端注册端口。
        ServerSocket serverSocket = new ServerSocket(8080);

        while (true) {
            // 2、使用serverSocket对象，调用一个accept方法，等待客户端的连接请求
            Socket socket = serverSocket.accept();

            System.out.println("有人上线了：" + socket.getRemoteSocketAddress());

            // 3、把这个客户端对应的socket通信管道，交给一个独立的线程负责处理。
            new ServerReaderThread(socket).start();
        }
    }
}













>>>>>>> d418ed5a0517a1c34a75d286fd5c685a031f6eb6
