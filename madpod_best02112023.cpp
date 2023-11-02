#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <math.h>
#include <set>
#include <random>
#include <chrono>
#include <deque>
using namespace std::chrono;
using namespace std;

#define PI 3.1415926535897932384626433832795


struct Vector2{
    float x;
    float y;
};


float distance(Vector2 v1,Vector2 v2){
    return sqrt((v1.x-v2.x)*(v1.x-v2.x) + (v1.y-v2.y)*(v1.y-v2.y));
}

float norme1(Vector2 v){
    return sqrt(v.x*v.x + v.y*v.y);
}

float pscal(Vector2 v, Vector2 v2){
    return v.x*v2.x + v.y*v2.y;
}

float get_angle(Vector2 a, Vector2 b){

    float na = sqrt(a.x*a.x + a.y*a.y);
    float nb = sqrt(b.x*b.x + b.y*b.y);

    float vec = a.x *b.y - b.x*a.y;

    vec /= na*nb;

    float theta = asin(vec) * 180 / PI;

    return theta;

}

float get_angle2(Vector2 pt, Vector2 p) {
    float d = distance(pt, p);
    float dx = (pt.x - p.x) / d;
    float dy = (pt.y - p.y) / d;

    // Trigonométrie simple. On multiplie par 180.0 / PI pour convertir en degré.
    float a = acos(dx) * 180.0 / PI;

    // Si le point qu'on veut est en dessus de nous, il faut décaler l'angle pour qu'il soit correct.
    if (dy < 0) {
        a = 360.0 - a;
    }

    return a;
}



class Player{
public:
    Player(){}
    int x;
    int y;
    Vector2 speed;
    float angle;
    Vector2 direction;
    float angletot=0;
    int check_point;
    int thrust;
    int act_check;
    int check_pass = 0;
    bool shield = false;
    Player(int x, int y){
            this->x = x;
            this->y = y;
            this->angle = 0;
    }



};

float getAngle(Vector2 pt, Player &p) {
    float d = distance(pt, {p.x, p.y});
    float dx = (pt.x - p.x) / d;
    float dy = (pt.y - p.y) / d;

    // Trigonométrie simple. On multiplie par 180.0 / PI pour convertir en degré.
    float a = acos(dx) * 180.0 / PI;

    // Si le point qu'on veut est en dessus de nous, il faut décaler l'angle pour qu'il soit correct.
    if (dy < 0) {
        a = 360.0 - a;
    }

    return a;
}

float diffAngle(Vector2 p, Player &pl) {
    float a = getAngle(p, pl);

    // Pour connaitre le sens le plus proche, il suffit de regarder dans les 2 sens et on garde le plus petit
    // Les opérateurs ternaires sont la uniquement pour éviter l'utilisation d'un operateur % qui serait plus lent
    float right = pl.angletot <= a ? a - pl.angletot : 360.0 - pl.angletot + a;
    float left = pl.angletot >= a ? pl.angletot - a : pl.angletot + 360.0 - a;

    if (right < left) {
        return right;
    } else {
        // On donne un angle négatif s'il faut tourner à gauche
        return -left;
    }
}

void rotate(Vector2 p, Player &pl) {
    pl.angle = diffAngle(p, pl);

    // On ne peut pas tourner de plus de 18° en un seul tour
    if (pl.angle  > 18.0) {
        pl.angle  = 18.0;
    } else if (pl.angle  < -18.0) {
        pl.angle  = -18.0;
    }

    pl.angletot += pl.angle;

    // L'opérateur % est lent. Si on peut l'éviter, c'est mieux.
    if (pl.angle >= 360.0) {
        pl.angletot = pl.angletot - 360.0;
    } else if (pl.angletot < 0.0) {
        pl.angletot += 360.0;
    }
}

class Sim{
public:
    Sim(){this->interm = true;}
    Vector2 pos;
    Vector2 speed;
    float angletot=0;
    float angle;
    int thrust;
    Vector2 direction;
    int check_point;
    int check_pass;
    bool interm ;
    int score=0;

};

class Solution {
public:
    int nb_turn;
    deque<Sim> vsol;
    bool empty = true;
    int turn = 0;
    Solution(){}
    Solution(int nbt) :nb_turn(nbt) {
        this->vsol.resize(nb_turn);
        nb_turn = nbt;
        turn = 0;

        for (int i = 0; i < nb_turn; ++i) {
            Sim sm;
            sm.angletot = 0;
            sm.speed = { 10000, 10000 };
            sm.direction = { 0,0 };
            sm.pos = { 10000, 10000 };
            sm.score = -1000000000;
            sm.thrust = 200;
            vsol[i] = sm;
        }



    }

    void Init(Player p){
        turn = 0;

        for (int i = 0; i < nb_turn; ++i) {
            
            vsol[i] .angletot = p.angletot;
            vsol[i] .speed = {0, 0};
            vsol[i] .direction = { 0,0 };
            vsol[i] .pos = {0, 0};
            vsol[i] .score = -1000000000;
            //vsol[i] .angle = vsol[i].angle;
            //vsol[i] .thrust = vsol[i].thrust;
            //vsol[i] .check_point = p.check_point;
            
        }
    }

    void pop() {
        std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<int> dthrust(0, 200);
        std::uniform_int_distribution<int> dthrust2(20, 100);
        std::uniform_int_distribution<int> dangle(0, 18);
        std::uniform_int_distribution<int> danglesg(0, 1);

        vsol.pop_front();
        Sim sm;
        //sm.angletot = ((int)vsol[nb_turn-1].angletot + (int)vsol[nb_turn-1].angle + 360)%360;
        //sm.speed = vsol[nb_turn-1].speed;
        //sm.direction = { 0,0 };
        //sm.pos = vsol[nb_turn-1].pos;
        sm.score = -1000000000;
        //sm.check_point = vsol[nb_turn-1].check_point;
        sm.angle = dangle(rng) ;
        if(sm.angle < -18)sm.angle = -18;
        if(sm.angle > 18)sm.angle = 18;
        //int div = (abs(sm.angle) / 18.0f) * 4.0f;
        //if(div ==0)div = 1;
        //int u = 
        sm.thrust =  dthrust(rng);
        if(sm.thrust < 0)sm.thrust = 0;
        if(sm.thrust > 200)sm.thrust = 200;

        vsol.push_back(sm);
    }

    int next() {
        turn = (turn + 1) % nb_turn;
        return turn;
    }



};

class AdvancedAnnihilationHeterogen {
public:
    int sz_gen;
    int nb_turn;
    int nb_solution;
    vector<Vector2>checkpoints;
    Solution sol;
    vector<Sim> population;
    std::chrono::time_point<std::chrono::high_resolution_clock> startm;
    AdvancedAnnihilationHeterogen(int szg, int nbt, int nbs): sz_gen(szg), nb_turn(nbt), nb_solution(nbs) {
        sol = Solution(nbt);

        std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<int> dthrust(0, 200);
        std::uniform_int_distribution<int> dthrust2(20, 100);
        std::uniform_int_distribution<int> dangle(-18, 18);
        std::uniform_int_distribution<int> danglesg(0, 1);

        for(int i = 0;i < sz_gen;++i){
            Sim sm;
         
            sm.angle = dangle(rng) ;
            if(sm.angle < -18)sm.angle = -18;
            if(sm.angle > 18)sm.angle = 18;
            //int div =(abs(sm.angle) / 18.0f) * 4.0f;
            //if(div ==0)div = 1;
            sm.thrust = dthrust(rng);
            if(sm.thrust < 0)sm.thrust = 0;
            if(sm.thrust > 200)sm.thrust = 200;
            //sm.thrust /=   div;
            sm.score = -1000000000;
            population.push_back(sm);
        }

    }

    void Simulate(Sim& sim) {
        
        float anglef = int(sim.angletot + sim.angle + 360)% 360;
        //cerr << "angle " << sim.angletot << endl;

        float angleRad = anglef * PI / 180.f;
        sim.direction = { cos(angleRad) * (float)sim.thrust, sin(angleRad) * (float)sim.thrust };
        sim.speed = { sim.speed.x + sim.direction.x, sim.speed.y + sim.direction.y };
        sim.pos.x += int(sim.speed.x);
        sim.pos.y += int(sim.speed.y);
        sim.speed = { floor(sim.speed.x * 0.85f), floor(sim.speed.y * 0.85f) };


    }

    void Mutate(){

        std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<int> dthrust(0, 200);
        std::uniform_int_distribution<int> dthdiv(1, 4);
        std::uniform_int_distribution<int> dangle(-18, 18);
        std::uniform_int_distribution<int> dwhat(0, 2);
        std::uniform_int_distribution<int> dwho(0, sz_gen/4);
        std::uniform_int_distribution<int> danglesg(0, 1);

        //for(int i = 0;i < 5;++i){
        
            int i = dwho(rng);
        
          
            int wt = dwhat(rng);
            if (wt == 0){
                population[i].angle = dangle(rng);
                if(population[i].angle < -18)population[i].angle = -18;
                if(population[i].angle > 18)population[i].angle = 18;
            }
            else if(wt == 1){
                population[i].thrust = dthrust(rng);
                if(population[i].thrust < 0)population[i].thrust = 0;
                if(population[i].thrust > 200)population[i].thrust = 200;
            }
            else{
                population[i].angle = dangle(rng);
                //int div = (abs(population[i].angle) / 18.0f) * 4.0f;
                
                if(population[i].angle < -18)population[i].angle = -18;
                if(population[i].angle > 18)population[i].angle = 18;
                population[i].thrust = dthrust(rng);
                if(population[i].thrust < 0)population[i].thrust = 0;
                if(population[i].thrust > 200)population[i].thrust = 200;
                //if(div ==0)div = 1;
                //population[i].thrust /= div;

            }
        //}

    }

    void Cross(){

        std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<int> dthrust(0, 200);
        std::uniform_int_distribution<int> dthdiv(1, 4);
        std::uniform_int_distribution<int> dangle(-18, 18);
        std::uniform_int_distribution<int> dwhat(0, 1);
        std::uniform_int_distribution<int> dwho(3, sz_gen/2);
        std::uniform_int_distribution<int> danglesg(0, 1);

        for(int i = 0;i < sz_gen/8;++i){

            int who = i;
            while(who == i)
                who = dwho(rng);

            int wt = dwhat(rng);
            if (wt == 0){
                population[i].angle = population[who].angle;
           
            }
            else if(wt == 1){
                population[i].thrust = population[who].thrust;
            

            }
        }

    }

    void Kill(){

        std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<int> dthrust(0, 200);
        std::uniform_int_distribution<int> dthdiv(1, 4);
        std::uniform_int_distribution<int> dangle(-18, 18);
        std::uniform_int_distribution<int> dwhat(0, 1);
        std::uniform_int_distribution<int> danglesg(0, 1);

        for(int i = sz_gen*9/10;i < sz_gen;++i){
         
            population[i].angle = dangle(rng);
            population[i].thrust = dthrust(rng);
            
            if(population[i].angle < -18)population[i].angle = -18;
            if(population[i].angle > 18)population[i].angle = 18;
            if(population[i].thrust < 0)population[i].thrust = 0;
            if(population[i].thrust > 200)population[i].thrust = 200;
            //int div = (abs(population[i].angle) / 18.0f) * 4.0f;
            //if(div ==0)div = 1;
            //population[i].thrust /= div;
    
            population[i].score = -1000000000;

        }


    }


    void CreateSolution(int time, Player p, bool first) {

        std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<int> dthrust(0, 200);
        
        std::uniform_int_distribution<int> dthrust2(20, 100);
        std::uniform_int_distribution<int> dangle(0, 18);
        std::uniform_int_distribution<int> danglesg(0, 1);
        std::uniform_int_distribution<int> dpop(0, this->sz_gen-1);
        this->startm = high_resolution_clock::now();
        auto getTime = [&]()-> bool {
            auto stop = high_resolution_clock::now();
            auto duration = duration_cast<milliseconds>(stop - this->startm);
            return(duration.count() <= time);
        };


        sol.pop();
        sol.Init(p);

        Sim smf ;
        smf.thrust = 200;
        smf.angle = 0;

        //sol.vsol[0].check_point = p.check_point;

        //for(int i = 0;i < population.size();++i)population[i].score = 1000000000;
        
        bool start = true;
        sol.turn = 0;
        int count = 0;
        int ind = 0;
        int check_pass_t = 1000000;
        int maxsteer = -10000000;
        while (getTime()) {

            //cerr << "turn " << sol.turn << endl;
            Sim sm;

        
            if (sol.turn > 0) {
                sm.angletot = ((int)sol.vsol[sol.turn-1].angletot + (int)sol.vsol[sol.turn-1].angle + 360)%360;
                sm.speed.x = sol.vsol[sol.turn-1].speed.x;
                sm.speed.y = sol.vsol[sol.turn-1].speed.y;
                sm.direction = { 0,0 };
                sm.pos = sol.vsol[sol.turn-1].pos;
                sm.score = sol.vsol[sol.turn-1].score;
                sm.check_point = sol.vsol[sol.turn - 1].check_point;
                sm.check_pass = sol.vsol[sol.turn - 1].check_pass;
                //sm.angle = sol.vsol[sol.turn - 1].angle;
                //ind = dpop(rng);
                //sm.thrust = population[ind].thrust ;
            }
            else {
                sm.angletot = p.angletot;
                sm.speed.x = p.speed.x;
                sm.speed.y = p.speed.y;
                sm.direction = { 0,0 };
                sm.pos = { (float)p.x, (float)p.y };
                sm.score = 1000000000;
                sm.check_point = p.check_point;
                sm.check_pass = p.check_pass;
                                

            }
            
            ind = dpop(rng);
            sm.angle = population[ind].angle;
            sm.thrust = population[ind].thrust ;
            
            

            //cerr << 1 << endl;

            
            Simulate(sm);

            //cerr << 2 << endl;
            int prec = ((sm.check_point - 1) + checkpoints.size()) %  checkpoints.size();
             int prec2 = ((sm.check_point - 2) + checkpoints.size()) %  checkpoints.size();
             int next1 = (sm.check_point + 1) % checkpoints.size();
            Vector2 vprec = {this->checkpoints[sm.check_point].x - this->checkpoints[prec].x, this->checkpoints[sm.check_point].y - this->checkpoints[prec].y};
            Vector2 vnext = {this->checkpoints[next1].x - this->checkpoints[prec].x, this->checkpoints[next1].y - this->checkpoints[prec].y};
            float angle_act = get_angle(vprec, vnext);
            float angle_act2 = get_angle2(vprec, vnext);
            float pscal_act = pscal(vprec, vnext);

            Vector2 vprecch = {this->checkpoints[sm.check_point].x - this->checkpoints[prec].x, this->checkpoints[sm.check_point].y - this->checkpoints[prec].y};
            Vector2 vusch = {this->checkpoints[sm.check_point].x - sm.pos.x, this->checkpoints[sm.check_point].y - sm.pos.y};
            
            float angle_check = get_angle(vprecch, vusch);

            //if(angle_act < 0 && angle_act2 > 90 && angle_act2 <= 170)angle_act = angle_act2;
         
            double d = distance(this->checkpoints[sm.check_point], sm.pos);
            int chk = 1;
            if (d <= 600) {
                
                check_pass_t = min(check_pass_t, sol.turn);
                smf.thrust = 0;
                chk++;
                sm.check_pass++;
                sm.check_point = (sm.check_point + 1) % this->checkpoints.size();
            }

            float ds = 1000;
            Vector2 dv = {this->checkpoints[sm.check_point].x - sm.pos.x, this->checkpoints[sm.check_point].y - sm.pos.y};
            float nrm = norme1(dv);
            dv.x = (dv.x / nrm) * ds;
            dv.y = (dv.y / nrm) * ds;

            float nrms = norme1(sm.speed);
            Vector2 dsp = {sm.speed.x / nrms, sm.speed.y / nrms};
            dsp.x *= ds;
            dsp.y *= ds;

            Vector2 steering = {dv.x - dsp.x, dv.y - dsp.y};
            float angle = abs(get_angle(dv, sm.speed));
            int speed = norme1(steering);
            float vspeed = norme1(sm.speed);

            sm.score = sm.check_pass * 30000  - d -speed ;
            //sm.score = (float)sm.score *  (1.0f - (angle+0.000001) / 90.0f);
                      
        
            if (sm.score > sol.vsol[sol.turn].score || start) {
                //cerr << "steering " <<  speed << endl;
                //cerr << angle_act << endl;
                if(sol.turn == 0)
                    maxsteer = speed;
                sol.vsol[sol.turn] = sm;
                sol.vsol[sol.turn].thrust = (float)sm.thrust * abs((1.0f - (speed) / 2000.0f));
                if(d <= (vspeed * 1500.0f / 600.f)  && abs(angle_act) >= 30){
                    sol.vsol[sol.turn].thrust = 0;

                }
                if(sol.vsol[sol.turn].thrust < 0) sol.vsol[sol.turn].thrust = 0;
                if(sol.vsol[sol.turn].thrust > 200) sol.vsol[sol.turn].thrust = 200;
                population[ind].score=max(population[ind].score, sm.score);
                
            }

            

            start = false;
            sol.next();
            ++count;
            if(count == sol.nb_turn){
                first = false;
            }

            if(count % this->sz_gen == 0){
                //cerr << "init" << endl;
                sort(population.begin(), population.end(), [](Sim a, Sim b) -> bool {
                    return a.score > b.score;
               });
                Mutate();
                Cross();
                Kill();
            }


        }

        /*float minid = -1000000;
        int indx = 0;
        for(int i = 0;i < sol.nb_turn;++i){
            double d = sol.vsol[i].score;
            if(d > minid){
                minid = d;
                indx = i;
            }

        }

        sol.vsol[0] = sol.vsol[ind];*/

        cerr << "count " << count << " " << maxsteer << endl;
       


    }

    void CreateSolutionOp(int time, Player &p, Player popp) {

        std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<int> dthrust(0, 200);
        
        std::uniform_int_distribution<int> dthrust2(20, 100);
        std::uniform_int_distribution<int> dangle(0, 18);
        std::uniform_int_distribution<int> danglesg(0, 1);
        std::uniform_int_distribution<int> dpop(0, this->sz_gen-1);
        this->startm = high_resolution_clock::now();
        auto getTime = [&]()-> bool {
            auto stop = high_resolution_clock::now();
            auto duration = duration_cast<milliseconds>(stop - this->startm);
            return(duration.count() <= time);
        };


        sol.pop();
        sol.Init(p);

        Sim smf ;
        smf.thrust = 200;
        smf.angle = 0;

        //sol.vsol[0].check_point = p.check_point;

        //for(int i = 0;i < population.size();++i)population[i].score = 1000000000;
        
        bool start = true;
        sol.turn = 0;
        int count = 0;
        int ind = 0;
        int check_pass_t = 1000000;
        int maxsteer = -10000000;
        while (getTime()) {

            //cerr << "turn " << sol.turn << endl;
            Sim sm;

        
            if (sol.turn > 0) {
                sm.angletot = ((int)sol.vsol[sol.turn-1].angletot + (int)sol.vsol[sol.turn-1].angle + 360)%360;
                sm.speed.x = sol.vsol[sol.turn-1].speed.x;
                sm.speed.y = sol.vsol[sol.turn-1].speed.y;
                sm.direction = { 0,0 };
                sm.pos = sol.vsol[sol.turn-1].pos;
                sm.score = sol.vsol[sol.turn-1].score;
                sm.check_point = sol.vsol[sol.turn - 1].check_point;
                sm.check_pass = sol.vsol[sol.turn - 1].check_pass;
                //sm.angle = sol.vsol[sol.turn - 1].angle;
                //ind = dpop(rng);
                //sm.thrust = population[ind].thrust ;
            }
            else {
                sm.angletot = p.angletot;
                sm.speed.x = p.speed.x;
                sm.speed.y = p.speed.y;
                sm.direction = { 0,0 };
                sm.pos = { (float)p.x, (float)p.y };
                sm.score = 1000000000;
                sm.check_point = p.check_point;
                sm.check_pass = p.check_pass;
                                

            }
            
            ind = dpop(rng);
            sm.angle = population[ind].angle;
            sm.thrust = population[ind].thrust ;
            
            

            //cerr << 1 << endl;

            
            Simulate(sm);

            //cerr << 2 << endl;
            int prec = ((sm.check_point - 1) + checkpoints.size()) %  checkpoints.size();
             int prec2 = ((sm.check_point - 2) + checkpoints.size()) %  checkpoints.size();
             int next1 = (sm.check_point + 1) % checkpoints.size();
            Vector2 vprec = {this->checkpoints[sm.check_point].x - this->checkpoints[prec].x, this->checkpoints[sm.check_point].y - this->checkpoints[prec].y};
            Vector2 vnext = {this->checkpoints[next1].x - this->checkpoints[prec].x, this->checkpoints[next1].y - this->checkpoints[prec].y};
            float angle_act = get_angle(vprec, vnext);
            float angle_act2 = get_angle2(vprec, vnext);
            float pscal_act = pscal(vprec, vnext);

            Vector2 vprecch = {this->checkpoints[sm.check_point].x - this->checkpoints[prec].x, this->checkpoints[sm.check_point].y - this->checkpoints[prec].y};
            Vector2 vusch = {this->checkpoints[sm.check_point].x - sm.pos.x, this->checkpoints[sm.check_point].y - sm.pos.y};
            
            float angle_check = get_angle(vprecch, vusch);

            //if(angle_act < 0 && angle_act2 > 90 && angle_act2 <= 170)angle_act = angle_act2;
         
            double d = distance({popp.x, popp.y}, sm.pos);
            int chk = 1;
            if (d <= 600) {

                if(sol.turn < 2){
                    p.shield = true;
                }
                
                check_pass_t = min(check_pass_t, sol.turn);
                smf.thrust = 0;
                chk++;
                sm.check_pass++;
                //sm.check_point = (sm.check_point + 1) % this->checkpoints.size();
            }

            float ds = 1000;
            Vector2 dv = {popp.x - sm.pos.x, popp.y - sm.pos.y};
            float nrm = norme1(dv);
            dv.x = (dv.x / nrm) * ds;
            dv.y = (dv.y / nrm) * ds;

            float nrms = norme1(sm.speed);
            Vector2 dsp = {sm.speed.x / nrms, sm.speed.y / nrms};
            dsp.x *= ds;
            dsp.y *= ds;

            Vector2 steering = {dv.x - dsp.x, dv.y - dsp.y};
            float angle = abs(get_angle(dv, sm.speed));
            int speed = norme1(steering);
            float vspeed = norme1(sm.speed);

            sm.score = sm.check_pass * 30000  - d -speed ;
            //sm.score = (float)sm.score *  (1.0f - (angle+0.000001) / 90.0f);
                      
        
            if (sm.score > sol.vsol[sol.turn].score || start) {
                //cerr << "steering " <<  speed << endl;
                //cerr << angle_act << endl;
                if(sol.turn == 0)
                    maxsteer = speed;
                sol.vsol[sol.turn] = sm;
                //sol.vsol[sol.turn].thrust = (float)sm.thrust * abs((1.0f - (speed) / 2000.0f));
                //if(d <= (vspeed * 1500.0f / 600.f)  && abs(angle_act) >= 30){
                //    sol.vsol[sol.turn].thrust = 0;

                //}
                if(sol.vsol[sol.turn].thrust < 0) sol.vsol[sol.turn].thrust = 0;
                if(sol.vsol[sol.turn].thrust > 200) sol.vsol[sol.turn].thrust = 200;
                population[ind].score=max(population[ind].score, sm.score);
                
            }

            

            start = false;
            sol.next();
            ++count;
           
            if(count % this->sz_gen == 0){
                //cerr << "init" << endl;
                sort(population.begin(), population.end(), [](Sim a, Sim b) -> bool {
                    return a.score > b.score;
               });
                Mutate();
                Cross();
                Kill();
            }


        }
    
        cerr << "count " << count << " " << maxsteer << endl;
       
    }



};

class FuckUp{
public:
    FuckUp(){

    }
    int depth;
    int w;
    int sz_gen;
    vector<Vector2>checkpoints;
    vector<Sim> sim;
    int minstartd;
    int ncheck;
    bool fcheck;
    bool depart = true;
    int fthrust;
    int angletotal=0;
    int checkpass = 0;
    bool collision = false;
    int turn_col = 0;
    bool indrift = false;
    int turn_drift = 0;
    bool repos = false;
    int angle_prec = 0;
    int angle_precrace = 0;
    std::chrono::time_point<std::chrono::high_resolution_clock> startm;
    FuckUp(int depth, int w, int szg){
        this->depth = depth;
        this->w = w;
        this->sz_gen = szg;
        minstartd = -1;

        std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<int> dthrust(0, 100);
        std::uniform_int_distribution<int> dangle(0, 18);
        std::uniform_int_distribution<int> danglesg(0, 1);

        for(int i = 0;i < szg;++i){
            Sim sm;
            int sg = 1;
            int s = danglesg(rng);
            if (s == 1)sg = -1;
            sm.angle = dangle(rng) * sg;
            sm.thrust = dthrust(rng);
            //cerr << sm.angle << " " << sm.thrust << endl; 
            sim.push_back(sm);
        }
        //cerr <<"----------------"<<endl;

    }

    void Simulate(Sim &sim){
        /*if (inf)
            sim.angletot = int((sim.angletot  + sim.angle)-angle) % 360;
        else*/
         sim.angletot = int(sim.angletot  + sim.angle) % 360;
        //cerr << "angle " << sim.angletot << endl;

        float angleRad = sim.angletot  * PI / 180.f;
        sim.direction = {cos(angleRad) * (float)sim.thrust, sin(angleRad) * (float)sim.thrust};
        sim.speed = {sim.speed.x + sim.direction.x, sim.speed.y + sim.direction.y};
        sim.pos.x += int(sim.speed.x);
        sim.pos.y += int(sim.speed.y);
        /*if(sim.pos.x > 16000){
            sim.pos.x = 16000;            
        }
        if(sim.pos.x < 0){
            sim.pos.x = 0;            
        }
        if(sim.pos.y > 9000){
            sim.pos.y = 9000;            
        }
        if(sim.pos.y < 0){
            sim.pos.y = 0;            
        }*/
        sim.speed = {floor(sim.speed.x * 0.85f), floor(sim.speed.y * 0.85f)};
      

    }

    void Simulate2(Sim &sim){

        sim.angletot = int(sim.angletot  + sim.angle) % 180;
        //cerr << "angle " << sim.angletot << endl;

        float angleRad = sim.angletot  * PI / 180.f;
        sim.direction = {cos(angleRad) * (float)sim.thrust, sin(angleRad) * (float)sim.thrust};
        
        sim.pos.x += sim.direction.x;
        sim.pos.y += sim.direction.y;


    }

    void InitSim(int x, int y){

        std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<int> dthrust(0, 200);
        std::uniform_int_distribution<int> dangle(0, 36);
       
        Sim sm;
        /*for(int i = 0;i < this->sz_gen;++i){
            sm.check_point = 1;
            sm.pos = {x, y};
            sm.angle = dangle(rng)-18;
            sm.thrust = dthrust(rng);
            sim.push_back(sm);
            
        }*/

        //for(int i = -18;i <= 18;i+=4){
            for(int j = 5;j <= 20; j+=1){
                sm.check_point = 1;
                sm.pos = {x, y};
                //sm.angle = i;
                sm.thrust = j;
                sim.push_back(sm);
            }
        //}

        this->sz_gen = sim.size();


    }

    void Mutate(Player p){

        std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<int> dthrust(0, 100);
        std::uniform_int_distribution<int> dangle(0, 18);
        std::uniform_int_distribution<int> dwho(0, sim.size()-1);
        std::uniform_int_distribution<int> dwhat(0, 2);
        std::uniform_int_distribution<int> danglesg(0, 1);

        //for(int i = 0;i < sim.size()/2;++i){
            Sim sm;
            int ind = dwho(rng);
            sm = sim[ind];
            int sg = 1;
            int s = danglesg(rng);
            if (s == 1)sg = -1;
            int wt = dwhat(rng);
            if (wt == 0)
                sm.angle = dangle(rng);
            else if(wt == 1)
                sm.thrust = dthrust(rng);
            else{
                sm.angle = dangle(rng)*sg;
                sm.thrust = dthrust(rng);
            }



            sm.angletot = p.angletot;
            sm.speed = p.speed;
            sm.direction = {0,0};
            sm.pos = {p.x, p.y};
            sm.score = 0;
            Simulate(sm);

            double d = distance(this->checkpoints[p.check_point], sm.pos);
            int chk = checkpass;
            if(d <= 600)chk++;
            sm.score = 100000*chk - d;

            if(sm.score > sim[ind].score){
                sim[ind] = sm;
            }


        //}

        


    }

    void BS(Player &p, int time, int checkpass, Player p3 , Player p4){

        int sz =  sim.size();
        set<pair<float,int>> dist;

        vector<Vector2> v2nextcheck;
        

        auto getTime = [&]()-> bool {
            auto stop = high_resolution_clock::now();
            auto duration = duration_cast<milliseconds>(stop - this->startm);
            return(duration.count() <= time);
        };


        auto getTime2 = [&](std::chrono::time_point<std::chrono::high_resolution_clock> startt2)-> bool {
            auto stop = high_resolution_clock::now();
            auto duration = duration_cast<milliseconds>(stop - startt2);
            return(duration.count() <= time);
        };

        std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<int> dthrust(0, 200);
        std::uniform_int_distribution<int> dthrust2(20, 100);
        std::uniform_int_distribution<int> dangle(0, 18);
        std::uniform_int_distribution<int> dwho(0, this->sz_gen-1);
        std::uniform_int_distribution<int> danglesg(0, 1);

        double startd, mstartd;
        
      
        if(depart){
            int x=this->checkpoints[p.check_point].x, y = this->checkpoints[p.check_point].y;
            p.x = x;
            p.y = y;
            p.thrust = 200;
            depart = false;
            return;
        }
        
        //startd = distance({p.x, p.y}, this->checkpoints[p.check_point]);
        

        cerr <<p.check_point << endl;
        
        //startd += startd/ 2;
        
        int thrust = 0, mthrust;
        int x, y, mx, my;
        bool enter = false;
        x=0, y = 0;
        int angleinf = 0;
        float anglesel = 0;
        bool declenche = false;

    
       
             int prec = ((p.check_point - 1) + checkpoints.size()) %  checkpoints.size();
             int prec2 = ((p.check_point - 2) + checkpoints.size()) %  checkpoints.size();
             int next1 = (p.check_point + 1) % checkpoints.size();

            Vector2 vprec = {this->checkpoints[prec].x - this->checkpoints[p.check_point].x, this->checkpoints[prec].y - this->checkpoints[p.check_point].y};
            Vector2 vnext = {this->checkpoints[next1].x - this->checkpoints[p.check_point].x, this->checkpoints[next1].y - this->checkpoints[p.check_point].y};
            float angle_act = get_angle(vprec, vnext);
            float angle_act2 = get_angle2(vprec, vnext);
            float pscal_act = pscal(vprec, vnext);

            if(angle_act < 0 && angle_act2 > 90 && angle_act2 <= 170)angle_act = angle_act2;

            Vector2 vprec2 = {this->checkpoints[prec2].x - this->checkpoints[prec].x, this->checkpoints[prec2].y - this->checkpoints[prec].y};
            Vector2 vnext2 = {this->checkpoints[prec].x - this->checkpoints[p.check_point].x, this->checkpoints[prec].y - this->checkpoints[p.check_point].y};
            float angle_actp = get_angle(vprec2, vnext2);
            float angle_act2p = get_angle2(vprec2, vnext2);

            if(angle_actp < 0 && angle_act2p > 90 && angle_act2p <= 170)angle_actp = angle_act2p;

            Vector2 vprecch = {this->checkpoints[p.check_point].x - this->checkpoints[prec].x, this->checkpoints[p.check_point].y - this->checkpoints[prec].y};
            Vector2 vusch = {this->checkpoints[p.check_point].x - p.x, this->checkpoints[p.check_point].y - p.y};
            
            float angle_check = get_angle(vprecch, vusch);

            cerr << "angle_act " << angle_act << " anglecheck " << angle_act2 << " " << next1 << " " << prec << endl;
            cerr << "pscal " << pscal_act << endl;
            
                startd = distance(this->checkpoints[p.check_point], this->checkpoints[prec]);
//cerr << "angletot " << p.angletot << endl;
            mstartd = startd*2;
            int count = 0;
            int numcheck = 0;
            bool inf = false;
            bool sup = false;
            int ind = -1;
            int bestind = -1;
            double anglefinal = 0;
            bool enter2 = false;
            bool enter3 = false;
            while(getTime()){
                //cerr << 1 << endl;
                float ldprec = distance(this->checkpoints[prec2], this->checkpoints[prec]);
                float dprec = distance(this->checkpoints[p.check_point], this->checkpoints[prec]);
                float dp = distance(this->checkpoints[p.check_point], {p.x, p.y});

                Sim sm;
                sm.angletot = p.angletot;
                sm.speed = p.speed;
                sm.direction = {0,0};
                sm.pos = {p.x, p.y};
                sm.score = 0;

                int sg = 1;
                int s = danglesg(rng);
                if (s == 1)sg = -1;
                sm.angle = dangle(rng) * sg;

                
                

                if((  ( dp >= (dprec /2)+600)  || (indrift && turn_drift >= 10) )  && !repos){
                    
                    if(indrift && turn_drift >= 10){
                        repos = true;
                        continue;
                    }

                    sm.thrust = dthrust(rng);

                    Simulate(sm);

                    indrift = false;
                    turn_drift=0;

                    /******************************/
                    float ds = 1000;
                    Vector2 dv = {this->checkpoints[p.check_point].x - sm.pos.x, this->checkpoints[p.check_point].y - sm.pos.y};
                    float nrm = norme1(dv);
                    dv.x = (dv.x / nrm) * ds;
                    dv.y = (dv.y / nrm) * ds;

                    float nrms = norme1(sm.speed);
                    Vector2 dsp = {sm.speed.x / nrms, sm.speed.y / nrms};
                    dsp.x *= ds;
                    dsp.y *= ds;

                    Vector2 steering = {dv.x - dsp.x, dv.y - dsp.y};
                    int speed = norme1(steering);

                    double d = distance(this->checkpoints[p.check_point], sm.pos);
                    int chk = checkpass;
                    if(d <= 600)chk++;
                    sm.score = 100000*chk - d - speed;
                    /*********************************/

                    if(d < startd ){
                        //cerr << d << endl;
                        enter = true;
                        startd = d;//sm.score;
                        anglefinal = sm.angletot;

                        //cerr << "dprec " << dprec/2 << " dp " << dp << endl;

                        //double speed = norme1(sm.speed);
                        //cerr << "speed " << norme1(sm.speed) << endl;
                        //cerr <<dprec << endl;
                        //if(dp <= 3000 && speed >= 150  && (sm.angle >= 10 || sm.angle <= -10) )sm.thrust = (int)((dp / dprec) * sm.thrust)/2;
                        //if((angle_act < 50 && angle_act > -50) && dprec <= 7000)sm.thrust /= 2;
                        
                        //if(angle_act > 0 && angle_check < 0)anglefinal -= angle_check*10;
                        //else if(angle_act < 0 && angle_check > 0)anglefinal += angle_check*10;
                        //if(sm.thrust > 200)sm.thrust = 200;
                        
                        thrust = (float)sm.thrust * abs((1.0f - (speed) / 2000.0f));
                        if(thrust < 0)thrust = 0;
                        if(thrust > 200)thrust = 200;
                        
                        //anglefinal += angle_check;
                        //if (anglefinal < 0) anglefinal = ((int)anglefinal + 360)%360;
                        //else anglefinal = (int)anglefinal % 360;

                        float angleRad = anglefinal * PI / 180.f;
        
                        //Vector2 dir = {this->checkpoints[p.check_point].x - p.x, this->checkpoints[p.check_point].y - p.y};
                        Vector2 dir = {cos(angleRad) * 10000.0f, sin(angleRad) * 10000.0f};
                        x = dir.x;
                        y = dir.y;
                        x*=10000, y*=10000;
                        fthrust = thrust;

                    }
                   

                }
                else{

                    double d3 = distance({p.x, p.y}, {p3.x, p3.y});
                    double d4 = distance({p.x, p.y}, {p4.x, p4.y});

                    if(d3 <= 1200 || d4 <= 1200 ||collision ||repos){
                        int next;
                        next = p.check_point;
                        Vector2 dir;
                        dir = {this->checkpoints[next].x - p.x, this->checkpoints[next].y - p.y};
                        x = dir.x;
                        y = dir.y;
                        x*=10000, y*=10000;
                       
                       

                        double speed = norme1(sm.speed);
                        cerr << "speed " << speed << endl;
                        if(speed <= 400 && (sm.angle <= 5 &&  sm.angle >= -5))sm.thrust = 200;
                        else sm.thrust = (int)((dp / dprec) * 200.0f);
                        if(sm.thrust > 200)sm.thrust = 200;
                       
                        fthrust = sm.thrust;
                        collision = true;
                        repos = true;
                        indrift = false;
                        turn_drift=0;
                        enter3 = true;
                    }
                    else {

                        sm.thrust = (int)((dp / dprec) * 200.0f);
                        if(sm.thrust > 200)sm.thrust = 200;
                        int next, next2;
                        enter2 = true;
                        
                        int anglep = abs(angle_prec - p.angletot);

                        double speed = norme1(sm.speed);
                        if(speed >= 150 && dp <= 3000)sm.thrust /= 2;
                        if(speed <= 75 && dp <= 1500 && (sm.angle<=5 && sm.angle >= -5))sm.thrust *= 4;
                        if(sm.thrust > 200)sm.thrust = 200;

                        /*bool ent = false;
                        if((angle_act < 0 && angle_check > 0) || (angle_act > 0 && angle_check < 0))ent = true;

                        if(!indrift && !ent){
                            Sim smd = sm;
                            bool decl =false;
                            ///test declenche drift
                            for(int i = 0;i < 5;++i){
                                Simulate(smd);
                                double d = distance(this->checkpoints[p.check_point], smd.pos);
                                //double speed = norme1(smd.speed);
                                //if(speed >= 150 && d <= 3000)smd.thrust /= 2;
                                //if(speed <= 75 && d <= 1500 && (smd.angle<=5 && sm.angle >= -5))smd.thrust *= 4;
                                //if(smd.thrust > 200)smd.thrust = 200;
                                
                                if(d <= 600){
                                    decl = true;
                                    break;
                                }

                            }

                            if(decl){
                                next = (p.check_point + 1)% this->checkpoints.size();
                                next2 = (p.check_point + 2)% this->checkpoints.size();
                                declenche = true;
                                indrift = true;
                                turn_drift++;
                                
                            }
                            else{
                                next = p.check_point;
                        
                            }

                        }
                        else {
                            
                        }*/

                        Vector2 dir;

                        if ( (angle_act < 85 && angle_act >=0 && angle_actp > -85 && angle_actp < 0) ||
                             (angle_act > -85 && angle_act <0 && angle_actp < 85 && angle_actp >= 0) ){
                            next  =p.check_point;
                            indrift = false;
                            turn_drift=0;
                            sm.thrust= (float)(dp / dprec) * 200.0f;
                            if(sm.thrust > 200)sm.thrust = 200;
                            dir = {this->checkpoints[next].x - p.speed.x*5, this->checkpoints[next].y - p.speed.y * 5};
                            //dir = {this->checkpoints[next].x - p.x, this->checkpoints[next].y - p.y};
                            x = dir.x;
                            y = dir.y;
                        }
                        else{
                            next = (p.check_point + 1)% this->checkpoints.size();
                            cerr << "next " << next << endl;
                            indrift = true;
                            turn_drift++;
                            dir = {this->checkpoints[next].x - p.x, this->checkpoints[next].y - p.y};
                            x = dir.x;
                            y = dir.y;
                            x*=10000, y*=10000;
                        }


                        
                        

                        
                        
                        fthrust = sm.thrust;
                        startd = 1000000000;
                    }

                    break;

                }
                
               ++count;

            }

           cerr << enter << " " << enter2 << " " << enter3<< endl;
           cerr << "repos "<<repos  << " indrift " << indrift << " " << turn_drift<< endl;
            if(!enter && !enter2 && !enter3){
                cerr << "false" << endl;
                float dprec = distance(this->checkpoints[p.check_point], this->checkpoints[prec]);
                float dp = distance(this->checkpoints[p.check_point], {p.x, p.y});
                int anglep = abs(angle_prec - p.angletot);
                thrust =  200;//(dp / dprec) * 200.0f;
                if(anglep>=15 )thrust = thrust / 2;
                if(thrust > 200)thrust = 200;

                int next;
                indrift = false;
                turn_drift=0;
                next = p.check_point;
                Vector2 dir = {this->checkpoints[next].x - p.x, this->checkpoints[next].y - p.y};
                x = dir.x;
                y = dir.y;
                x*=10000, y*=10000;
                fthrust = thrust;
               

            }
            
            

            cerr << "BS " << count <<  " " << anglefinal << " " <<  thrust << endl;

        

        
        
        p.x = x;
        p.y = y;
        p.thrust = fthrust;


    }

    Player Play(int x, int y, int vx, int vy, int angle, int next_chk, int time, int checkpass, Player p3 , Player p4){
       
        checkpass = checkpass;
        Player p = Player(x, y);
        p.speed = {vx, vy};
        p.angletot = angle;
        //cerr << "angletot " << p.angletot << endl;
        p.check_point = next_chk;
        BS(p, time, checkpass, p3, p4);
        //Mutate();
        cerr << "return " << endl;
        return p;  

    }

};

int main()
{
    int laps;
    cin >> laps; cin.ignore();
    int checkpoint_count;
    vector<Vector2>checkpoint;
    cin >> checkpoint_count; cin.ignore();
    for (int i = 0; i < checkpoint_count; i++) {
        int checkpoint_x;
        int checkpoint_y;
        cin >> checkpoint_x >> checkpoint_y; cin.ignore();
        checkpoint.push_back({checkpoint_x, checkpoint_y});
    }

    FuckUp fuckup = FuckUp(1,10, 200);
    fuckup.checkpoints = checkpoint;
    //fuckup.InitSim(10353, 986);
    //auto startt = high_resolution_clock::now();
    fuckup.startm = high_resolution_clock::now();
    fuckup.ncheck = 0;
    fuckup.fcheck = false;
    fuckup.minstartd = 2000000;

    FuckUp fuckup2 = FuckUp(1,10, 200);
    fuckup2.checkpoints = checkpoint;
    //fuckup.InitSim(10353, 986);
    //auto startt = high_resolution_clock::now();
    fuckup2.startm = high_resolution_clock::now();
    fuckup2.ncheck = 0;
    fuckup2.fcheck = false;
    fuckup2.minstartd = 2000000;

    AdvancedAnnihilationHeterogen aah = AdvancedAnnihilationHeterogen(100, 5, 1);
    aah.checkpoints = checkpoint;

    AdvancedAnnihilationHeterogen aah2 = AdvancedAnnihilationHeterogen(100, 5, 1);
    aah2.checkpoints = checkpoint;

    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> dshield(0, 20);

    Player p1, p2, _p1, _p2, _p3, _p4;// = fuckup.Play(10353, 986, 10, 10, 0, 1, 500);
    _p1.act_check = 1;
    _p2.act_check = 1;
    _p1.check_pass++;
    _p2.check_pass++;
    _p3.act_check = 1;
    _p4.act_check = 1;
    _p3.check_pass++;
    _p4.check_pass++;
    bool start = true;
    // game loop
    while (1) {
        for (int i = 0; i < 2; i++) {
            int x; // x position of your pod
            int y; // y position of your pod
            int vx; // x speed of your pod
            int vy; // y speed of your pod
            int angle; // angle of your pod
            int next_check_point_id; // next check point id of your pod
            if(i == 0)fuckup.angle_prec = angle;
            else fuckup2.angle_prec = angle;
            cin >> x >> y >> vx >> vy >> angle >> next_check_point_id; cin.ignore();
            cerr << angle << endl;
            if(i == 0){
                _p1.x = x;
                _p1.y = y;
                _p1.speed.x = vx;
                _p1.speed.y = vy;
                _p1.angletot = angle;
                _p1.check_point = next_check_point_id;
                if(next_check_point_id != _p1.act_check){
                    _p1.act_check = next_check_point_id;
                    _p1.check_pass++;
                    //fuckup.collision = false;
                    //fuckup.repos = false;
                    
                }
            } 
            else{
                _p2.x = x;
                _p2.y = y;
                _p2.speed.x = vx;
                _p2.speed.y = vy;
                _p2.angletot = angle;
                _p2.check_point = next_check_point_id;
                if(next_check_point_id != _p2.act_check){
                    _p2.act_check = next_check_point_id;
                    _p2.check_pass++;
                    fuckup2.collision = false;
                    fuckup2.repos = false;
                }
            }

       

        

        // X Y THRUST MESSAGE

        }
        for (int i = 0; i < 2; i++) {
            int x; // x position of the opponent's pod
            int y; // y position of the opponent's pod
            int vx; // x speed of the opponent's pod
            int vy; // y speed of the opponent's pod
            int angle; // angle of the opponent's pod
            int next_check_point_id; // next check point id of the opponent's pod
            cin >> x >> y >> vx >> vy >> angle >> next_check_point_id; cin.ignore();

            if(i == 0){
                _p3.x = x;
                _p3.y = y;
                _p3.speed.x = vx;
                _p3.speed.y = vy;
                _p3.angletot = angle;
                _p3.check_point = next_check_point_id;
                if(next_check_point_id != _p3.act_check){
                    _p3.act_check = next_check_point_id;
                    _p3.check_pass++;
                }
            } 
            else{
                _p4.x = x;
                _p4.y = y;
                _p4.speed.x = vx;
                _p4.speed.y = vy;
                _p4.angletot = angle;
                _p4.check_point = next_check_point_id;
                if(next_check_point_id != _p4.act_check){
                    _p4.act_check = next_check_point_id;
                    _p4.check_pass++;
                }
            }


        }

        double d = distance(checkpoint[_p1.check_point], {_p1.x, _p1.y});
        int scorep1 = _p1.check_pass * 30000  - d;

        double d2 = distance(checkpoint[_p2.check_point], {_p2.x, _p2.y});
        int scorep2 = _p2.check_pass * 30000  - d2;

        double d3 = distance(checkpoint[_p3.check_point], {_p3.x, _p3.y});
        int scorep3 = _p3.check_pass * 30000  - d3;

        double d4 = distance(checkpoint[_p4.check_point], {_p4.x, _p4.y});
        int scorep4 = _p4.check_pass * 30000  - d4;

        bool inchase = false;
        if(start){
                //fuckup.startm = high_resolution_clock::now();
                //p1 = fuckup.Play(_p1.x, _p1.y, _p1.speed.x, _p1.speed.y, _p1.angletot, _p1.check_point, 400, _p1.check_pass, _p3, _p4);
                Vector2 dv = {(checkpoint[_p1.check_point].x - _p1.x), (checkpoint[_p1.check_point].y - _p1.y)};
                _p1.speed = dv;
                _p1.check_point = 1;
                aah.CreateSolution(400, _p1, false);

                Vector2 dv2 = {_p3.x - _p2.x, _p3.y - _p2.y};
                _p2.speed = dv2;
                aah2.CreateSolutionOp(400, _p2, _p3);

                //fuckup2.startm = high_resolution_clock::now();
                //p2 = fuckup2.Play(_p2.x, _p2.y, _p2.speed.x, _p2.speed.y, _p2.angletot, _p2.check_point, 400, _p2.check_pass,  _p3, _p4);
        }
        else{
                //fuckup.startm = high_resolution_clock::now();
                //p1 = fuckup.Play(_p1.x, _p1.y, _p1.speed.x, _p1.speed.y, _p1.angletot, _p1.check_point, 25, _p1.check_pass,  _p3, _p4);
                if(scorep1 > scorep2){
                    inchase = false;
                    aah.CreateSolution(50, _p1, true);
                    if(scorep3 > scorep4)
                        aah2.CreateSolutionOp(20, _p2, _p3);
                    else
                        aah2.CreateSolutionOp(20, _p2, _p4);
                }
                else{
                    inchase = true;
                    aah2.CreateSolution(50, _p2, true);
                    if(scorep3 > scorep4)
                        aah.CreateSolutionOp(20, _p1, _p3);
                    else
                        aah.CreateSolutionOp(20, _p1, _p4);

                }
                //fuckup2.startm = high_resolution_clock::now();
                //p2 = fuckup2.Play(_p2.x, _p2.y, _p2.speed.x, _p2.speed.y, _p2.angletot, _p2.check_point, 10, _p2.check_pass,  _p3, _p4);

        }

        //cerr << p.x << " " << p.y << " "<< p.thrust << endl;
        //float anglef = int(aah.sol.vsol[0].angletot + aah.sol.vsol[0].angle + 360) % 360;
        //float angleRad = anglef * PI / 180.f;
        //Vector2 dir = {cos(angleRad) * 10000.0f, sin(angleRad) * 10000.0f};

        if(!inchase){
            int x = checkpoint[_p1.check_point].x - 3*aah.sol.vsol[0].speed.x;
            int y = checkpoint[_p1.check_point].y - 3*aah.sol.vsol[0].speed.y;
            //int x = _p1.x + dir.x;
            //int y = _p1.y + dir.y;
        
        
            int thrust = aah.sol.vsol[0].thrust;


        
            if(thrust > 200)thrust = 200;
            //x*=10000, y*=10000;
            if(start){
                cout << checkpoint[_p1.check_point].x << " " << checkpoint[_p1.check_point].y << " BOOST" << endl;
            }
            else{
                if(max(scorep1, max(scorep2, max(scorep3, scorep4))) == scorep1){
                    cout << x << " " << y << " " <<  thrust << " I get YOU !!!" << endl;

                }
                else{
                    cout << x << " " << y << " " <<  thrust << endl;
                }
            }

            float anglef = int(aah2.sol.vsol[0].angletot + aah2.sol.vsol[0].angle + 360) % 360;
            float angleRad = anglef * PI / 180.f;
            Vector2 dir = {cos(angleRad) * 10000.0f, sin(angleRad) * 10000.0f};
            int x2 = _p2.x + dir.x;
            int y2 = _p2.y + dir.y;

            //int x2 = _p4.x - 3*aah2.sol.vsol[0].speed.x;
            //int y2 = _p4.y - 3*aah2.sol.vsol[0].speed.y;
            int thrust2 = aah2.sol.vsol[0].thrust;
            
            if(start){
                cout << checkpoint[_p2.check_point].x << " " << checkpoint[_p2.check_point].y << " BOOST" << endl;
            }
            else{
                bool shield = false;
                if(_p2.shield){
                    if(dshield(rng)<5)shield = true;
                    _p2.shield = false;
                }
                if(max(scorep1, max(scorep2, max(scorep3, scorep4))) == scorep2){
                    if(!shield)
                        cout << x2 << " " << y2 << " " <<  thrust2 << " I get YOU !!!" << endl;
                    else
                        cout << x2 << " " << y2 << " " <<  "SHIELD SHIELD I get YOU !!!" << endl;

                }
                else{
                    if(!shield)
                        cout << x2 << " " << y2 << " " <<  thrust2 << endl;
                    else
                        cout << x2 << " " << y2 << " " <<  "SHIELD SHIELD" << endl;
                }
            }

        }
        else{

            float anglef = int(aah.sol.vsol[0].angletot + aah.sol.vsol[0].angle + 360) % 360;
            float angleRad = anglef * PI / 180.f;
            Vector2 dir = {cos(angleRad) * 10000.0f, sin(angleRad) * 10000.0f};
            int x = _p1.x + dir.x;
            int y = _p1.y + dir.y;
        
        
            int thrust = aah.sol.vsol[0].thrust;


        
            if(thrust > 200)thrust = 200;
            //x*=10000, y*=10000;
            if(start){
                cout << checkpoint[_p1.check_point].x << " " << checkpoint[_p1.check_point].y << " BOOST" << endl;
            }
            else{
                bool shield = false;
                if(_p1.shield){
                    if(dshield(rng)<5)shield = true;
                    _p1.shield = false;
                }

                if(max(scorep1, max(scorep2, max(scorep3, scorep4))) == scorep1){
                    if(!shield)
                        cout << x << " " << y << " " <<  thrust << " I get YOU !!!" << endl;
                    else
                        cout << x << " " << y << " " <<  "SHIELD SHIELD I get YOU !!!" << endl;
                }
                else{
                    if(!shield)
                        cout << x << " " << y << " " <<  thrust << endl;
                    else
                        cout << x << " " << y << " SHIELD SHIELD" << endl;
                }
            }

            int x2 = checkpoint[_p2.check_point].x - 3*aah2.sol.vsol[0].speed.x;
            int y2 = checkpoint[_p2.check_point].y - 3*aah2.sol.vsol[0].speed.y;

            int thrust2 = aah2.sol.vsol[0].thrust;
            
            if(start){
                cout << checkpoint[_p2.check_point].x << " " << checkpoint[_p2.check_point].y << " BOOST" << endl;
            }
            else{

                if(max(scorep1, max(scorep2, max(scorep3, scorep4))) == scorep2){
                    cout << x2 << " " << y2 << " " <<  thrust2 << " I get YOU !!!" << endl;

                }
                else{
                    cout << x2 << " " << y2 << " " <<  thrust2 << endl;
                }
            }



        }

        start = false;
        // Write an action using cout. DON'T FORGET THE "<< endl"
        // To debug: cerr << "Debug messages..." << endl;


        // You have to output the target position
        // followed by the power (0 <= power <= 200)
        // i.e.: "x y power"
        //cout << "8000 4500 100" << endl;
        //cout << "8000 4500 100" << endl;
    }
}