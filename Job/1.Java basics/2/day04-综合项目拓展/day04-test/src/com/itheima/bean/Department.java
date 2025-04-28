<<<<<<< HEAD
package com.itheima.bean;

import java.util.ArrayList;
// 科室类
public class Department {
    private String name;
    private ArrayList<Doctor> doctors = new ArrayList<>();

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public ArrayList<Doctor> getDoctors() {
        return doctors;
    }

    public void setDoctors(ArrayList<Doctor> doctors) {
        this.doctors = doctors;
    }

    public int getNumber() {
        return doctors.size();
    }
}
=======
package com.itheima.bean;

import java.util.ArrayList;
// 科室类
public class Department {
    private String name;
    private ArrayList<Doctor> doctors = new ArrayList<>();

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public ArrayList<Doctor> getDoctors() {
        return doctors;
    }

    public void setDoctors(ArrayList<Doctor> doctors) {
        this.doctors = doctors;
    }

    public int getNumber() {
        return doctors.size();
    }
}
>>>>>>> d418ed5a0517a1c34a75d286fd5c685a031f6eb6
