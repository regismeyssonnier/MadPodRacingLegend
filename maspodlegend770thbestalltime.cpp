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

#define ll long long int

struct Vector2{
    float x;
    float y;
    Vector2(){};
    Vector2(float xx, float yy):x(xx), y(yy){};
    
};


float norme1(Vector2 v){
    return sqrt(v.x*v.x + v.y*v.y);
}

float dot(const Vector2 &a, const Vector2 &b) {
  return a.x*b.x+a.y*b.y;
}
float cross(const Vector2 &vec, const Vector2 &axe) {
	//projeté de vec sur la direction orthogonale à axe, à +90°
  return vec.y*axe.x-vec.x*axe.y;
}

float crossproduct(Vector2 a, Vector2 b){
    return a.x * b.y - a.y * a.x;
}

float getAngleA(Vector2 a, Vector2 b){
    float cp = crossproduct(a, b);
    float an =  atan2(abs(cp), dot(a, b)) * 180.0 / PI;
    //if(cp < 0)an = 360 - an;
    return an;
}

float getAngleA2(Vector2 a, Vector2 b){
    float n1 = norme1(a);
    float n2 = norme1(b);

    return acos(dot(a, b) / (n1*n2)) * 180.0 / PI;


}

Vector2 vec2ByAngle(float angle){
    return Vector2(cos(angle), sin(angle));
}

float distance(Vector2 v1,Vector2 v2){
    return sqrt((v1.x-v2.x)*(v1.x-v2.x) + (v1.y-v2.y)*(v1.y-v2.y));
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
    bool check_stay = false;
    bool attack_stay = false;
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
    Vector2 pos, pos2;
    Vector2 speed, speed2;
    float angletot=0, angletot2;
    float angle;
    int thrust;
    Vector2 direction;
    int check_point, check_point2;
    int check_pass, check_pass2;
    bool interm ;
    ll score=0;
    int x;
    int y;
    bool shield = false;
    bool shield2 = false;
    int check_stay = -1;
    bool attack_stay = false;
    bool in_position = false;
    bool hunt  =false;
    Vector2 final_point;
    vector<double> gene = vector<double>(6);
    int leader = 0;
    int opp = 0;
    
    
};


class _5G{
public:
    int NB_SIM;
    int NB_POP;
    int DEPTH;
    vector<Sim> next_gen;
    vector<vector<Sim>> population;
    vector<Vector2>checkpoints;
    float podRadius = 400.f;
    float podRadiusSqr = podRadius * podRadius;

    _5G(){}
    _5G(int nb_sim, int nb_pop, int d): NB_SIM(nb_sim), NB_POP(nb_pop), DEPTH(d){
        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<double> dgene(0, 1.0);

        for (int i = 0; i < DEPTH; ++i) {
            population.push_back({});
            while (population[i].size() < NB_POP) {
                Sim sm;
            
                for(int j = 0;j < sm.gene.size();++j){
                    sm.gene[j] = dgene(rng);
                    
                }

                sm.score = -1000000000;
                population[i].push_back(sm);
            }
        }

    }

    void Selection(int depth){
        next_gen = {};
        double sz = (double)population[depth].size()*0.3;
        for(int i = 0;i < sz;++i){
            next_gen.push_back(population[depth][i]);

        }

        std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<int> dng(sz, population[depth].size()-1);

        double sz2 = (double)population[depth].size()*0.2;
        for(int i = 0;i<sz2;++i ){
            next_gen.push_back(population[depth][dng(rng)]);
        }
       

    }

    void NextGen2(int depth){

        Selection(depth);

        vector<Sim> children;

        std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<int> dng(0, next_gen.size()-1);
        std::uniform_real_distribution<double> dgene(0, 1.0);
        std::uniform_int_distribution<int> dcross(0, 3);
                
        int nb_mutate = 0;
        int MAX_MUT = next_gen.size();
        while(children.size() < next_gen.size()){

            Sim child;
         
            for(int i = 0;i < 2;++i){
                Sim par1 = next_gen[dng(rng)];
                Sim par2 = next_gen[dng(rng)];

            
                child.gene[i*3] = (par1.gene[i*3] + par2.gene[i*3]) / 2.0;
                child.gene[i*3+1] = (par1.gene[i*3+1] + par2.gene[i*3+1]) / 2.0;

                child.score = -1000000000;

            }

            if(nb_mutate < MAX_MUT){

                for(int i = 0;i < 2;++i){
              
                    int cross = dcross(rng) ;
                    
                    if(cross < 3){
                        child.gene[i*3+cross] = dgene(rng);
                    }
                    else{
                        for(int j = 0;j < 2;++j){
                            child.gene[i*3+j] = dgene(rng);
                        }
                    }

                }

                ++nb_mutate;

            }

            children.push_back(child);

        }

        next_gen.insert(next_gen.end(), children.begin(), children.end());

        population[depth].swap(next_gen);


    }

    void Decalage_gen2(){
        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<double> dgene(0, 1.0);

        for(int i = 0;i<  DEPTH - 1;++i){
            population[i] = population[i+1];
        }
       
        for(int i = 0;i < NB_POP;++i){
            Sim sm;
                   
            for(int j = 0;j < sm.gene.size();++j){
                sm.gene[j] = dgene(rng);
            }

            sm.score = -1000000000;
  
            population[DEPTH-1][i] = sm;
            
        }
        

    }

    void Simulate(Sim& sim) {

        float anglef = int(sim.angletot + sim.angle + 360) % 360;
        //cerr << "angle " << sim.angletot << endl;

        float angleRad = anglef * PI / 180.f;
        sim.direction = { cos(angleRad) * (float)sim.thrust, sin(angleRad) * (float)sim.thrust };
        sim.speed = { sim.speed.x + sim.direction.x, sim.speed.y + sim.direction.y };
        sim.pos.x += int(sim.speed.x);
        sim.pos.y += int(sim.speed.y);
        sim.speed = { floor(sim.speed.x * 0.85f), floor(sim.speed.y * 0.85f) };
        //cerr << "end" << endl;

    }

    void InitSim(Sim &pod1, int depth, int ind, int starti){

        if(population[depth][ind].gene[starti+2] > 0.95){
            pod1.shield = true;

        }
       
        double gr = population[depth][ind].gene[starti+0];
        if(gr < 0.25)
            pod1.angle = -18;
        else if(gr > 0.25)
            pod1.angle = 18;
        else
            pod1.angle = 18.0 * ((gr - 0.25) * 2.0);

        double gt = population[depth][ind].gene[starti+1];
        if(gt < 0.25)
            pod1.thrust = 0;
        else if(gt > 0.75)
            pod1.thrust = 200;
        else
            pod1.thrust = 200.0 * ((gt - 0.25) * 2.0);

        

    }

    vector<string> PlaySim(Player p, Player p2, Player p3, Player p4, int turn, int time){

        auto startm = high_resolution_clock::now();;
        int maxt = -1;
        auto getTime = [&]()-> bool {
            auto stop = high_resolution_clock::now();
            auto duration = duration_cast<milliseconds>(stop - startm);
            //cerr << duration.count() << endl;
            maxt = max(maxt, (int)duration.count());
            return(duration.count() <= time);
        };

        if(DEPTH > 1 && turn > 0){
            Decalage_gen2();
        }

        int ind = 0;
        int depth = 0;
        int nb_turn = 0;
        int nb_sim = 0;
        int leader2 = 0;

        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<double> dgene(0, 1.0);

        vector<vector<Sim>> pod(DEPTH, vector<Sim>(2));
        {
            Sim sm;
            sm.angletot = p.angletot;
            sm.speed.x = p.speed.x;
            sm.speed.y = p.speed.y;
            sm.direction = { 0,0 };
            sm.pos = { (float)p.x, (float)p.y };
            sm.score = -1000000000;
            sm.check_point = p.check_point;
            sm.check_pass = p.check_pass;
            pod[0][0] = sm;

            Sim sm2;
            sm2.angletot = p2.angletot;
            sm2.speed.x = p2.speed.x;
            sm2.speed.y = p2.speed.y;
            sm2.direction = { 0,0 };
            sm2.pos = { (float)p2.x, (float)p2.y };
            sm2.score = -1000000000;
            sm2.check_point = p2.check_point;
            sm2.check_pass = p2.check_pass;
            pod[0][1] = sm2;

        }

        

        cerr << "start " << endl;
        while(getTime()){

            Sim pod1, pod2;
            

            if(depth == 0){
                pod1 = pod[0][0];
                pod2 = pod[0][1];
            }
            else{
                pod1 = population[depth-1][0];
                pod2.pos = population[depth-1][0].pos2;
                pod2.speed = population[depth-1][0].speed2;
                pod2.check_point = population[depth-1][0].check_point2;
                pod2.check_pass = population[depth-1][0].check_pass2;
                pod2.angletot = population[depth-1][0].angletot2;
                pod2.check_stay = population[depth-1][0].check_stay;
            }
            
            InitSim(pod1, depth, ind, 0);
            InitSim(pod2, depth, ind, 3);

            //cerr << "init " << endl;

            Simulate(pod1);
            Simulate(pod2);

            ////cerr << "simulate " << endl;

            int score = 0;

            Vector2 checkp1=checkpoints[pod1.check_point];
            /*double g = dgene(rng);
            checkp1.x += cos(g * 2.0 * PI) * 600.0;
            checkp1.y += sin(g * 2.0 * PI) * 600.0;*/

            int p1dist = distance(pod1.pos, checkp1);
            int p2dist = distance(pod2.pos, checkpoints[pod2.check_point]);
            int p3dist = distance({p3.x, p3.y}, checkpoints[p3.check_point]);
            int p4dist = distance({p4.x,p4.y}, checkpoints[p4.check_point]);

            //cerr << "simulate2 " << endl;


            int leader = 0;

            if(p1dist <= 600){
                pod1.check_pass++;
                pod1.check_point = (pod1.check_point + 1) % this->checkpoints.size();
            }
            if(p2dist <= 600){
                pod2.check_pass++;
                pod2.check_point = (pod2.check_point + 1) % this->checkpoints.size();
            }
            //cerr << "simulate 3 " << endl;

            int scorep1 = 50000 * pod1.check_pass - p1dist;
            int scorep2 = 50000 * pod2.check_pass - p2dist;
            int scorep3 = 50000 * p3.check_pass - p3dist;
            int scorep4 = 50000 * p4.check_pass - p4dist;

            if(scorep1 > scorep2)leader = 1;
            else leader = 2;

             Sim follow;
            if(scorep3 > scorep4){
                follow.speed = p3.speed;
                follow.pos = {p3.x, p3.y};
                follow.check_point = p3.check_point;
                follow.score = scorep3;
                leader = 3;
            }
            else{
                follow.speed = p4.speed;
                follow.pos = {p4.x, p4.y};
                follow.check_point = p4.check_point;
                follow.score = scorep4;
                leader = 4;
            }


            //cerr << "distance " << endl;

            pod1.shield = 0;
            pod2.shield = 0;

            Player pl1;
            pl1.angletot = int(pod1.angletot + 360 + pod1.angle) % 360;

            score += scorep1-follow.score;
            double a = diffAngle(this->checkpoints[pod1.check_point], pl1);            
            if (a > 18.0) {
                a = 18.0;
            } else if (a < -18.0) {
                a = -18.0;
            }

            score += -(pod1.angle-a);
            //score += scorep2;

            float ds = 1000;
            Vector2 dv = {this->checkpoints[pod1.check_point].x - pod1.pos.x, this->checkpoints[pod1.check_point].y - pod1.pos.y};
            float nrm = norme1(dv);
            dv.x = (dv.x / nrm) * ds;
            dv.y = (dv.y / nrm) * ds;

            float nrms = norme1(pod1.speed);
            Vector2 dsp = {pod1.speed.x / nrms, pod1.speed.y / nrms};
            dsp.x *= ds;
            dsp.y *= ds;

            Vector2 steering = {dv.x - dsp.x, dv.y - dsp.y};
            double spsteering = norme1(steering);

            int nextcheck = (pod1.check_point + 1) % checkpoints.size();
            double x1 = pod1.pos.x;
            double y1 = pod1.pos.y;
            double x2 = this->checkpoints[pod1.check_point].x;
            double y2 = this->checkpoints[pod1.check_point].y;
            double x3 = this->checkpoints[nextcheck].x;
            double y3 = this->checkpoints[nextcheck].y;
            double angle = atan2(y1-y2,x1-x2)-atan2(y3-y2,x3-x2);
            angle = angle * 180.0 / PI;
            angle = fmod((angle + 180.0), 360.0);
            if (angle < 0.0)
                angle += 360.0;
            angle -= 180.0;
            double anglech = atan2(y2 - y1, x2 - x1);
            anglech = anglech * 180.0 / PI;
            // Ajustement par rapport à l'angle total (p.angletot)
            anglech = fmod(anglech - pod1.angletot + 540, 360) - 180;

            int nextch = (pod1.check_point+1)%checkpoints.size();
            int p1dist2 = distance(pod1.pos, checkpoints[nextch]);

            double nd = norme1(p3.speed);
            Vector2 sp = p3.speed;
            sp.x /= nd;
            sp.y /= nd;
                            
            Vector2 posr = {p3.x + sp.x * 600, p3.y - sp.y * 600};
           

            double col = double((p3.speed.x * (pod2.pos.x - p3.x) + p3.speed.y*(pod2.pos.y-p3.y))) / 
            double(sqrt(p3.speed.x*p3.speed.x + p3.speed.y*p3.speed.y) * 
                sqrt((pod2.pos.x - p3.x)* (pod2.pos.x - p3.x) +  (pod2.pos.y - p3.y) * (pod2.pos.y - p3.y))+0.000001);

            //if(col > 0.5){
                Player pl;
                pl.angletot = int(pod2.angletot + 360 + pod2.angle) % 360;
                score += -diffAngle({p3.x, p3.y}, pl);
                score += -distance(this->checkpoints[p3.check_point], {p3.x, p3.y});
                pod2.opp = 1;
            
            
            /*}
            else{
                 int nextch = (p3.check_point+1)%checkpoints.size();
                 if(pod2.check_stay == -1){
                    pod2.check_stay = 2;
                }
                int p3dist = distance(pod2.pos, checkpoints[2]);
                pod2.opp = 0;
                score += 50000-p3dist;

            }*/

            //score += -((180-anglech)*(1000.0)) * (double)pod1.thrust;
            
            //score -= (scorep3/2 + scorep4/2);

            /*if(population[depth][ind].gene[2] < 0.3 && scorep1 > scorep2){
                score += scorep1;
            }
            else if(population[depth][ind].gene[2] < 0.3 && scorep1 < scorep2){
               
                double nd = norme1(follow.speed);
                Vector2 sp = follow.speed;
                sp.x /= nd;
                sp.y /= nd;
                               
                Vector2 posr = {follow.pos.x + sp.x * 10, follow.pos.y - sp.y * 10};
                score += -distance(posr, pod1.pos);

            }
            else if(population[depth][ind].gene[2] < 0.2 && scorep1 < scorep2){
                score += -distance(pod1.pos, this->checkpoints[follow.check_point]);
            }
            else if(population[depth][ind].gene[2] < 0.1 && scorep1 < scorep2){
                int nnext = (follow.check_point + 1) % checkpoints.size();
                score += -distance(pod1.pos, this->checkpoints[nnext]);    
            }
            
            pod2.shield = 0;
            if(population[depth][ind].gene[5] < 0.3 && scorep1 < scorep2){
                score += scorep2;
            }
            else if(population[depth][ind].gene[5] < 0.3 && scorep1 > scorep2){
                double nd = norme1(follow.speed);
                Vector2 sp = follow.speed;
                sp.x /= nd;
                sp.y /= nd;
                               
                Vector2 posr = {follow.pos.x + sp.x * 10, follow.pos.y - sp.y * 10};
                score += -distance(posr, pod2.pos);
            }
            else if(population[depth][ind].gene[5] < 0.2 && scorep1 > scorep2){
                score += -distance(pod2.pos, this->checkpoints[follow.check_point]);
            }
            else if(population[depth][ind].gene[5] < 0.1 && scorep1 > scorep2){
                int nnext = (follow.check_point + 1) % checkpoints.size();
                score += -distance(pod2.pos, this->checkpoints[nnext]);    
            }*/

            //cerr << "strategy " << endl;
            population[depth][ind].leader = leader;
            population[depth][ind].opp = pod2.opp;
            population[depth][ind].pos = pod1.pos;
            population[depth][ind].pos2 = pod2.pos;
            population[depth][ind].speed = pod1.speed;
            population[depth][ind].speed2 = pod2.speed;
            population[depth][ind].check_point = pod1.check_point;
            population[depth][ind].check_point2 = pod2.check_point;
            population[depth][ind].check_pass = pod1.check_pass;
            population[depth][ind].check_pass2 = pod2.check_pass;
            population[depth][ind].angletot = pod1.angletot;
            population[depth][ind].angletot2 = pod2.angletot;
            population[depth][ind].check_stay = pod2.check_stay;

            population[depth][ind].shield = pod1.shield;
            population[depth][ind].shield2 = pod2.shield;
            population[depth][ind].score = score;


            nb_sim++;
            if (nb_sim == NB_SIM) {
                nb_sim = 0;

                
                
                sort(population[depth].begin(), population[depth].end(), [](Sim a, Sim b) -> bool {
                    return a.score > b.score;
                    });
                
                this->NextGen2(depth);

                                               

                depth++;
                if (depth == DEPTH) {
                  
                    depth = 0;
                }
            }

                           

            ind = (ind + 1) % NB_POP;
            nb_turn++;

        }

        cerr << "end " << endl;

        vector<string> ans;
        int x, y , x2, y2;
        Sim ps = population[0][0];

        //x = this->checkpoints[ps.check_point].x - ps.speed.x*3;
        //y = this->checkpoints[ps.check_point].y - ps.speed.y*3;
        //x2 = this->checkpoints[ps.check_point2].x - ps.speed2.x*3;
        //y2 = this->checkpoints[ps.check_point2].y - ps.speed2.y*3;

        double angle = 0;
                double gr = ps.gene[0];
            if(gr < 0.25)
                angle = -18;
            else if(gr > 0.75)
                angle = 18;
            else
                angle = 18.0 * ((gr - 0.25) * 2.0);

            float anglef = int(p.angletot + angle + 360) % 360;
            //cerr << "angle " << sim.angletot << endl;

            float angleRad = anglef * PI / 180.f;
            Vector2 dir = { cos(angleRad) * 10000.0f, sin(angleRad) * 10000.0f };
            x = p.x + dir.x;
            y = p.y + dir.y;

        double gt = population[0][0].gene[1];
        int thrust = 0;
        if(gt < 0.25)
            thrust = 0;
        else if(gt > 0.75)
            thrust = 200;
        else
            thrust = 200.0 * ((gt - 0.25) * 2.0);

        gt = population[0][0].gene[4];
        int thrust2 = 0;
        if(gt < 0.25)
            thrust2 = 0;
        else if(gt > 0.75)
            thrust2 = 200;
        else
            thrust2 = 200.0 * ((gt - 0.25) * 2.0);


        /*double angle = 0;
                double gr = ps.gene[3];
            if(gr < 0.25)
                angle = -18;
            else if(gr > 0.75)
                angle = 18;
            else
                angle = 18.0 * ((gr - 0.25) * 2.0);

            float anglef = int(p2.angletot + angle + 360) % 360;
            //cerr << "angle " << sim.angletot << endl;

            float angleRad = anglef * PI / 180.f;
            Vector2 dir = { cos(angleRad) * (float)1000, sin(angleRad) * (float)1000 };*/

            //ps.speed2 = { p2.speed.x + dir.x, p2.speed.y + dir.y };
            //sim.pos.x += int(ps.speed.x);
            //sim.pos.y += int(ps.speed.y);
            //x2 = p2.x + dir.x;
            //y2 = p2.y + dir.y;


        //if(ps.opp == 1){
            double nd = norme1(p3.speed)+0.000001;
                Vector2 sp = p3.speed;
                sp.x /= nd;
                sp.y /= nd;
                               
                Vector2 posr = {p3.x + sp.x * 600, p3.y - sp.y * 600};
            x2 = posr.x - ps.speed2.x*3;
            y2 = posr.y - ps.speed2.y*3;
            //thrust2 = max(thrust, 100);
        /*}
        else{
            
            int next = p3.check_stay;
            x2 = this->checkpoints[next].x - 3 *ps.speed2.x;
            y2 = this->checkpoints[next].y - 3 *ps.speed2.y;
        }*/

        /*if(ps.leader == 1){
            x = this->checkpoints[ps.check_point].x - ps.speed.x*3;
            y = this->checkpoints[ps.check_point].y - ps.speed.y*3;

          
                       
            {   
            double angle = 0;
                double gr = ps.gene[3];
            if(gr < 0.25)
                angle = -18;
            else if(gr > 0.75)
                angle = 18;
            else
                angle = 18.0 * ((gr - 0.25) * 2.0);

            float anglef = int(ps.angletot2 + angle + 360) % 360;
            //cerr << "angle " << sim.angletot << endl;

            float angleRad = anglef * PI / 180.f;
            Vector2 dir = { cos(angleRad) * 10000.0f, sin(angleRad) * 10000.0f };
            x2 = p2.x + dir.x;
            y2 = p2.y + dir.y;
            }
            
            
        }
        else{
            x2 = this->checkpoints[ps.check_point2].x - ps.speed2.x*3;
            y2 = this->checkpoints[ps.check_point2].y - ps.speed2.y*3;
           
        
            double angle = 0;
                double gr = ps.gene[0];
            if(gr < 0.25)
                angle = -18;
            else if(gr > 0.75)
                angle = 18;
            else
                angle = 18.0 * ((gr - 0.25) * 2.0);

            float anglef = int(ps.angletot2 + angle + 360) % 360;
            //cerr << "angle " << sim.angletot << endl;

            float angleRad = anglef * PI / 180.f;
            Vector2 dir = { cos(angleRad) * 10000.0f, sin(angleRad) * 10000.0f };
            x = p.x + dir.x;
            y = p.y + dir.y;
            
        }*/

        


        if(ps.shield == 0){
            ans.push_back(to_string(x) + " " + to_string(y) + " " + to_string(thrust));
        }
        else{
            ans.push_back(to_string(x) + " " + to_string(y) + " SHIELD");
        }

        if(ps.shield2 == 0){
            ans.push_back(to_string(x2) + " " + to_string(y2) + " " + to_string(thrust2));
        }
        else{
            ans.push_back(to_string(x2) + " " + to_string(y2) + " SHIELD");
        }

        return ans;


    }

    int MiniMax(Sim pod, int depth, int alpha, int beta, bool minimax,
     int score_cum, double &pt_nuis){
        
        //cerr << indc << " " << depth << endl;
        if(depth == 0){
            return score_cum;

        }

        this->Simulate(pod);
        //cerr << "score " << wscore << endl;
        double d = distance(checkpoints[pod.check_point], pod.pos);
        if(d <= 600){
            pod.check_pass++;
            pod.check_point = (pod.check_point + 1) % this->checkpoints.size();
        }
        score_cum += 50000 * pod.check_pass - d;
        //if(wscore != 0)return wscore;



        int score = 0;
        int ind = 0;

        if(minimax){
            score = -2000000000;

           
            for(int i = 0;i < 7;++i){
                if( i == 6){
                    pod.thrust = 50;
                    pod.angle = -18;
                }
                if( i == 5){
                    pod.thrust = 100;
                    pod.angle = -13;
                }
                if( i == 4){
                    pod.thrust = 150;
                    pod.angle = -8;
                }
                if( i == 3){
                    pod.thrust = 200;
                    pod.angle = 0;
                }
                if( i == 2){
                    pod.thrust = 150;
                    pod.angle = 8;
                }
                if( i == 1){
                    pod.thrust = 100;
                    pod.angle = 13;
                }
                if( i == 0){
                    pod.thrust = 50;
                    pod.angle = 18;
                }

                Sim po = pod;                        
                int s = this->MiniMax(po, depth-1, alpha, beta, !minimax, score_cum, pt_nuis);
                score = max(score, s);

                if (score >= beta) return score;

                alpha = max(alpha, score);

                    

                
            }

            return score;
            
        }
        else{
            score = 2000000000;

           
            for(int i = 0;i < 7;++i){
                if( i == 6){
                    pod.thrust = 50;
                    pod.angle = -18;
                }
                if( i == 5){
                    pod.thrust = 100;
                    pod.angle = -13;
                }
                if( i == 4){
                    pod.thrust = 150;
                    pod.angle = -8;
                }
                if( i == 3){
                    pod.thrust = 200;
                    pod.angle = 0;
                }
                if( i == 2){
                    pod.thrust = 150;
                    pod.angle = 8;
                }
                if( i == 1){
                    pod.thrust = 100;
                    pod.angle = 13;
                }
                if( i == 0){
                    pod.thrust = 50;
                    pod.angle = 18;
                }

                Sim po = pod;    
                int s = this->MiniMax(po, depth-1, alpha, beta, !minimax, score_cum, pt_nuis);
                score = min(score, s);

                if (score <= alpha) return score;

                beta = min(beta, score);

                    

                
            }

            return score;



        }



        return score;

    }

    float CollisionTime(Sim& p1, Sim& p2)
    {
        const Vector2 dP{p2.pos.x - p1.pos.x, p2.pos.y - p1.pos.y};
        const Vector2 dS{p2.speed.x - p1.speed.x, p2.speed.y - p1.speed.y};

        constexpr float eps = 0.000001f; // float precision...

        // we're looking for t such that:
        // |            p2(t)           -           p1(t)           | < 2*podRadius
        // |(p2.position + t*p2.speed)  - (p1.position + t*p1.speed)| < 2*podRadius
        // |(p2.position - p1.position) -  t*(p2.speed - p1.speed)  | < 2*podRadius
        // |         dP                 +            t*dS           | < 2*podRadius
        // t^2 dS^2 + t 2dPdS + dP^2 < 4 podRadius^2
        // t^2  a   + t   b   +   c      = 0;

        const float a = dot(dS, dS);
        if (a < eps) // moving away from each other
            return INFINITY;

        const float b = -2.f*dot(dP, dS);
        const float c = dot(dP,dP) - 4.f*podRadiusSqr;

        const float delta = b*b - 4.f*a*c;
        if (delta < 0.f) // no solution
            return INFINITY;

        const float t = (b - sqrt(delta)) / (2.f * a);
        if (t <= eps)
            return INFINITY;

        return t;
    }


};


class Solutionm {
public:
    Sim moves1[20];
    Sim moves2[20];
    ll score;
    int shield=0;
    Sim pod1;
    Sim pod2;
    
};

class Simulation{
public:
    int NB_SOL;
    int DEPTH;
    vector<Solutionm> solution ;
    vector<Vector2>checkpoints;
    float podRadius = 400.f;
    float podRadiusSqr = podRadius * podRadius;
    float minImpulse = 120.f;
    float frictionFactor = .85f;
    int MAXT = 400;
    int MINT = -100;
    int MAXA = 40;
    int MINA = -40;
    int state_chaser = 0, state_chaser2 = 0;

    Simulation(int nbs, int d):NB_SOL(nbs), DEPTH(d){
        std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<int> dthrust(MINT, MAXT);
        std::uniform_int_distribution<int> dangle(MINA, MAXA);

       
        for(int i = 0;i < nbs;++i){
            Solutionm sol = Solutionm();
            for(int j = 0;j < DEPTH;++j){
                sol.moves1[i].angle = dangle(rng);
                if(sol.moves1[i].angle < -18)sol.moves1[i].angle= -18;
                if(sol.moves1[i].angle > 18)sol.moves1[i].angle= 18;
                sol.moves2[i].angle = dangle(rng);
                if(sol.moves2[i].angle < -18)sol.moves2[i].angle= -18;
                if(sol.moves2[i].angle > 18)sol.moves2[i].angle= 18;

                sol.moves1[i].thrust = dthrust(rng);
                if(sol.moves1[i].thrust < 0)sol.moves1[i].thrust= 0;
                if(sol.moves1[i].thrust > 200)sol.moves1[i].thrust= 200;
                sol.moves2[i].thrust = dthrust(rng);
                if(sol.moves2[i].thrust < 0)sol.moves2[i].thrust= 0;
                if(sol.moves2[i].thrust > 200)sol.moves2[i].thrust= 200;

                sol.score = -2000000000;
            }

            solution.push_back(sol);

        }



    }

    void Trier(){

        sort(this->solution.begin(), this->solution.end(), [](Solutionm a, Solutionm b) -> bool {
            return a.score > b.score;
        });

    }

    void Simulate(Sim& sim) {

        float anglef = int(sim.angletot + sim.angle + 360) % 360;
        //cerr << "angle " << sim.angletot << endl;

        float angleRad = anglef * PI / 180.f;
        sim.direction = { cos(angleRad) * (float)sim.thrust, sin(angleRad) * (float)sim.thrust };
        sim.speed = { sim.speed.x + sim.direction.x, sim.speed.y + sim.direction.y };
        sim.pos.x += sim.speed.x;
        sim.pos.y += sim.speed.y;
        //sim.speed = { floor(sim.speed.x * 0.85f), floor(sim.speed.y * 0.85f) };
        //cerr << "end" << endl;

    }

    void EndSimulate(Sim &sim){
        sim.pos.x = int(sim.pos.x);
        sim.pos.y = int(sim.pos.y);
        sim.speed = { floor(sim.speed.x * 0.85f), floor(sim.speed.y * 0.85f) };

    }

    Solutionm Mutate(int ind, double amplitude){

        std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<int> dthrust(0, 200);
        std::uniform_int_distribution<int> dangle(-18, 18);
        std::uniform_int_distribution<int> dcross(0, 2);

        Solutionm sol = solution[ind];
        

        for(int i = 0;i < DEPTH;++i){
           
            float angle = sol.moves1[i].angle; 
            float ramin = angle - 36.0 * amplitude;
            float ramax = angle + 36.0 * amplitude;

            if (ramin < -18.0) {
                ramin = -18.0;
            }

            if (ramax > 18.0) {
                ramax = 18.0;
            }

            std::uniform_int_distribution<int> dang(ramin, ramax);
            angle = dang(rng);

            sol.moves1[i].angle = angle;

            angle = sol.moves2[i].angle; 
            ramin = angle - 36.0 * amplitude;
            ramax = angle + 36.0 * amplitude;

            if (ramin < -18.0) {
                ramin = -18.0;
            }

            if (ramax > 18.0) {
                ramax = 18.0;
            }

            std::uniform_int_distribution<int> dang2(ramin, ramax);
            angle = dang2(rng);

            sol.moves2[i].angle = angle;


        ///---------------------
            float thrust =sol.moves1[i].thrust;
            float pmin = thrust - 200 * amplitude;
            float  pmax = thrust + 200 * amplitude;

            if (pmin < 0) {
                pmin = 0;
            }

            if (pmax > 0) {
                pmax = 200;
            }

            std::uniform_int_distribution<int> dth(pmin, pmax);
            thrust = dth(rng);
            sol.moves1[i].thrust = thrust;
            
            thrust =sol.moves2[i].thrust;
            pmin = thrust - 200 * amplitude;
             pmax = thrust + 200 * amplitude;

            if (pmin < 0) {
                pmin = 0;
            }

            if (pmax > 0) {
                pmax = 200;
            }

            std::uniform_int_distribution<int> dth2(pmin, pmax);
            thrust = dth(rng);
            sol.moves2[i].thrust = thrust;
            

        }

        return sol;

    }

    Solutionm Mutate2(int ind, double amplitude){

        std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<int> dthrust(MINT, MAXT);
        std::uniform_int_distribution<int> dangle(MINA, MAXA);
        std::uniform_int_distribution<int> dcross(0, 2);

        Solutionm sol = solution[ind];

        for(int i = 0;i < DEPTH;++i){
            int num = dcross(rng);
            if(num  ==0){
                sol.moves1[i].angle = dangle(rng);
                if(sol.moves1[i].angle < -18)sol.moves1[i].angle = -18;
                if(sol.moves1[i].angle > 18 )sol.moves1[i].angle = 18;
                sol.moves2[i].angle = dangle(rng);
                if(sol.moves2[i].angle < -18)sol.moves2[i].angle = -18;
                if(sol.moves2[i].angle > 18 )sol.moves2[i].angle = 18;
            }
            else if(num ==1){
                sol.moves1[i].thrust = dthrust(rng);
                if(sol.moves1[i].thrust < 0)sol.moves1[i].thrust = 0;
                if(sol.moves1[i].thrust > 200 )sol.moves1[i].thrust = 200;

                sol.moves2[i].thrust = dthrust(rng);
                if(sol.moves2[i].thrust < 0)sol.moves2[i].thrust = 0;
                if(sol.moves2[i].thrust > 200 )sol.moves2[i].thrust = 200;
            }
            else {
                sol.moves1[i].angle = dangle(rng);
                if(sol.moves1[i].angle < -18)sol.moves1[i].angle = -18;
                if(sol.moves1[i].angle > 18 )sol.moves1[i].angle = 18;
                sol.moves2[i].angle = dangle(rng);
                if(sol.moves2[i].angle < -18)sol.moves2[i].angle = -18;
                if(sol.moves2[i].angle > 18 )sol.moves2[i].angle = 18;

                sol.moves1[i].thrust = dthrust(rng);
                if(sol.moves1[i].thrust < 0)sol.moves1[i].thrust = 0;
                if(sol.moves1[i].thrust > 200 )sol.moves1[i].thrust = 200;

                sol.moves2[i].thrust = dthrust(rng);
                if(sol.moves2[i].thrust < 0)sol.moves2[i].thrust = 0;
                if(sol.moves2[i].thrust > 200 )sol.moves2[i].thrust = 200;
            }
        }

        return sol;

    }

    void calc(Sim sm, double &col, double &angle, double &anglech){
         col = double((sm.speed.x * (this->checkpoints[sm.check_point].x - sm.pos.x) + sm.speed.y*(this->checkpoints[sm.check_point].y-sm.pos.y))) / 
            double(sqrt(sm.speed.x*sm.speed.x + sm.speed.y*sm.speed.y) * 
                sqrt((this->checkpoints[sm.check_point].x - sm.pos.x)* (this->checkpoints[sm.check_point].x - sm.pos.x) +  (this->checkpoints[sm.check_point].y - sm.pos.y) * (this->checkpoints[sm.check_point].y - sm.pos.y))+0.000001);

            int nextcheck = (sm.check_point + 1) % this->checkpoints.size();

            double x1 = sm.pos.x;
            double y1 = sm.pos.y;
            double x2 = this->checkpoints[sm.check_point].x;
            double y2 = this->checkpoints[sm.check_point].y;
            double x3 = this->checkpoints[nextcheck].x;
            double y3 = this->checkpoints[nextcheck].y;
            angle = atan2(y1-y2,x1-x2)-atan2(y3-y2,x3-x2);
            angle = angle * 180.0 / PI;
            angle = fmod((angle + 180.0), 360.0);
            if (angle < 0.0)
                angle += 360.0;
            angle -= 180.0;
            anglech = atan2(y2 - y1, x2 - x1);
            anglech = anglech * 180.0 / PI;
            // Ajustement par rapport à l'angle total (p.angletot)
            anglech = fmod(anglech - sm.angletot + 540, 360) - 180;

    }

    vector<string> Play(Player p, Player p2,Player p3, Player p4, int turn, int time){

        auto startm = high_resolution_clock::now();;
        int maxt = -1;
        auto getTime = [&]()-> bool {
            auto stop = high_resolution_clock::now();
            auto duration = duration_cast<milliseconds>(stop - startm);
            //cerr << duration.count() << endl;
            maxt = duration.count();
            return(duration.count() <= time);
        };

        std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<int> dthrust(MINT, MAXT);
        std::uniform_int_distribution<int> dangle(MINA, MAXA);
        std::uniform_int_distribution<int> dsol(0, NB_SOL-1);
     
        std::uniform_real_distribution<double> dgene(0, 1.0);

        if(turn > 0){
            for(int n = 0;n < NB_SOL;++n){
                
                for(int d = 0;d<  DEPTH-1;++d){
                    solution[n].moves1[d] = solution[n].moves1[d+1];
                    solution[n].moves2[d] = solution[n].moves2[d+1];
                    solution[n].score = -2000000000;
                }

                int i = DEPTH-1;
                solution[n].moves1[i].angle = dangle(rng);
                if(solution[n].moves1[i].angle < -18)solution[n].moves1[i].angle= -18;
                if(solution[n].moves1[i].angle > 18)solution[n].moves1[i].angle= 18;
                solution[n].moves2[i].angle = dangle(rng);
                if(solution[n].moves2[i].angle < -18)solution[n].moves2[i].angle= -18;
                if(solution[n].moves2[i].angle > 18)solution[n].moves2[i].angle= 18;

                solution[n].moves1[i].thrust = dthrust(rng);
                if(solution[n].moves1[i].thrust < 0)solution[n].moves1[i].thrust= 0;
                if(solution[n].moves1[i].thrust > 200)solution[n].moves1[i].thrust= 200;
                solution[n].moves2[i].thrust = dthrust(rng);
                if(solution[n].moves2[i].thrust < 0)solution[n].moves2[i].thrust= 0;
                if(solution[n].moves2[i].thrust > 200)solution[n].moves2[i].thrust= 200;
                
            }

            /*Solutionm sol = Solutionm();
            for(int i = 0;i<  DEPTH-1;++i){
                
                sol.moves1[i].angle = dangle(rng);
                if(sol.moves1[i].angle < -18)sol.moves1[i].angle= -18;
                if(sol.moves1[i].angle > 18)sol.moves1[i].angle= 18;
                sol.moves2[i].angle = dangle(rng);
                if(sol.moves2[i].angle < -18)sol.moves2[i].angle= -18;
                if(sol.moves2[i].angle > 18)sol.moves2[i].angle= 18;

                sol.moves1[i].thrust = dthrust(rng);
                if(sol.moves1[i].thrust < 0)sol.moves1[i].thrust= 0;
                if(sol.moves1[i].thrust > 200)sol.moves1[i].thrust= 200;
                sol.moves2[i].thrust = dthrust(rng);
                if(sol.moves2[i].thrust < 0)sol.moves2[i].thrust= 0;
                if(sol.moves2[i].thrust > 200)sol.moves2[i].thrust= 200;

                sol.score = -2000000000;

            }

            solution[NB_SOL-1] = sol;*/


        }

        int p3dist = distance({p3.x, p3.y}, checkpoints[p3.check_point]);
        int p4dist = distance({p4.x, p4.y}, checkpoints[p4.check_point]);

        int scorep3 = 50000 * p3.check_pass - p3dist;
        int scorep4 = 50000 * p4.check_pass - p4dist;

        Sim runner, chaser;
        if(scorep3 > scorep4){
             Sim sm;
                sm.angletot = p3.angletot;
                sm.speed.x = p3.speed.x;
                sm.speed.y = p3.speed.y;
                sm.direction = { 0,0 };
                sm.pos = { (float)p3.x, (float)p3.y };
                sm.score = -2000000000;
                sm.check_point = p3.check_point;
                sm.check_pass = p3.check_pass;
                sm.score = scorep3;
                runner = sm;

                Sim sm2;
                sm2.angletot = p4.angletot;
                sm2.speed.x = p4.speed.x;
                sm2.speed.y = p4.speed.y;
                sm2.direction = { 0,0 };
                sm2.pos = { (float)p4.x, (float)p4.y };
                sm2.score = -2000000000;
                sm2.check_point = p4.check_point;
                sm2.check_pass = p4.check_pass;
                sm2.score = scorep4;
                chaser = sm2;
        }
        else{
             Sim sm2;
                sm2.angletot = p4.angletot;
                sm2.speed.x = p4.speed.x;
                sm2.speed.y = p4.speed.y;
                sm2.direction = { 0,0 };
                sm2.pos = { (float)p4.x, (float)p4.y };
                sm2.score = -2000000000;
                sm2.check_point = p4.check_point;
                sm2.check_pass = p4.check_pass;
                sm2.score = scorep4;
                runner = sm2;
            Sim sm;
                sm.angletot = p3.angletot;
                sm.speed.x = p3.speed.x;
                sm.speed.y = p3.speed.y;
                sm.direction = { 0,0 };
                sm.pos = { (float)p3.x, (float)p3.y };
                sm.score = -2000000000;
                sm.check_point = p3.check_point;
                sm.check_pass = p3.check_pass;
                sm.score = scorep3;
                chaser = sm;
        }

        

        cerr << "start" << endl;

        

        double amplitude = 1.0;

        Solutionm solres ;
        solres.score = -2000000000;

        Sim pods1, pods2;
            {
                Sim sm;
                sm.angletot = p.angletot;
                sm.speed.x = p.speed.x;
                sm.speed.y = p.speed.y;
                sm.direction = { 0,0 };
                sm.pos = { (float)p.x, (float)p.y };
                sm.score = -2000000000;
                sm.check_point = p.check_point;
                sm.check_pass = p.check_pass;
                pods1 = sm;

                Sim sm2;
                sm2.angletot = p2.angletot;
                sm2.speed.x = p2.speed.x;
                sm2.speed.y = p2.speed.y;
                sm2.direction = { 0,0 };
                sm2.pos = { (float)p2.x, (float)p2.y };
                sm2.score = -2000000000;
                sm2.check_point = p2.check_point;
                sm2.check_pass = p2.check_pass;
                pods2 = sm2;

            }

        bool isleader = false;

        {
            int p1dist = distance(pods1.pos, this->checkpoints[pods1.check_point]);
            int p2dist = distance(pods2.pos, this->checkpoints[pods2.check_point]);   
            
            int scorep1 = 50000 * pods1.check_pass - p1dist;
            int scorep2 = 50000 * pods2.check_pass - p2dist;

            if(scorep1 >= scorep2)isleader = true;
            else isleader = false;

        }

        int nb_turn = 0;;
        
        while(getTime()){

            int score_chaser = 0;

            for(int ind = 0;ind < NB_SOL;++ind){
            
                Solutionm solret = Mutate2(ind, amplitude);
                
                amplitude = 1.0 - (double)maxt/70.0;

                Sim pod1, pod2;
                {
                    Sim sm;
                    sm.angletot = p.angletot;
                    sm.speed.x = p.speed.x;
                    sm.speed.y = p.speed.y;
                    sm.direction = { 0,0 };
                    sm.pos = { (float)p.x, (float)p.y };
                    sm.score = -2000000000;
                    sm.check_point = p.check_point;
                    sm.check_pass = p.check_pass;
                    pod1 = sm;

                    Sim sm2;
                    sm2.angletot = p2.angletot;
                    sm2.speed.x = p2.speed.x;
                    sm2.speed.y = p2.speed.y;
                    sm2.direction = { 0,0 };
                    sm2.pos = { (float)p2.x, (float)p2.y };
                    sm2.score = -2000000000;
                    sm2.check_point = p2.check_point;
                    sm2.check_pass = p2.check_pass;
                    pod2 = sm2;

                }
                //cerr << "init " << endl;

                ll scorep1 = 0;
                ll scorep2 = 0;
                ll distp1=0, distp2=0;
                for(int i = 0;i <DEPTH;++i){
                    //cerr << i << endl;
                    
                
                    //if(isleader ||(!isleader && i <= 2)){
                        pod1.angle = solret.moves1[i].angle;
                        pod1.thrust = solret.moves1[i].thrust;
                        Simulate(pod1);
                        pod1.angletot = int(pod1.angletot + pod1.angle + 360) % 360;
                        
                    //}

                // if(!isleader ||(isleader && i <= 2)){
                        pod2.angle = solret.moves2[i].angle;
                        pod2.thrust = solret.moves2[i].thrust;
                        Simulate(pod2);
                        pod2.angletot = int(pod2.angletot + pod2.angle + 360) % 360;
                // }

                    

                    bool stop = false;
                    double t = this->CollisionTime(pod1, pod2);
                    if(t <= 1.0){
                        //pod1.pos.x += pod1.speed.x * t; 
                        //pod1.pos.y += pod1.speed.y * t; 
                    // if(isleader ||(!isleader && i <= 2)){
                        double ns = norme1(pod1.speed);
                        Vector2 ahead = pod1.speed;
                        ahead.x /= ns;
                        ahead.y /= ns;
                        ahead.x *= 6000.;

                        ahead.x += pod1.pos.x; 
                        ahead.y += pod1.pos.y;

                        Vector2 diff;
                        diff.x = ahead.x - pod2.pos.x;
                        diff.y = ahead.y - pod2.pos.y;

                        double na = norme1(diff);
                        diff.x /= na;
                        diff.y /= na;
                        diff.x *= 600;
                        diff.y *= 600;

                        pod1.pos.x += diff.x;
                        pod1.pos.y += diff.y;

                    // }

                        //if(!isleader ||(isleader && i <= 2)){
                        /*pod2.pos.x += pod2.speed.x * t; 
                        pod2.pos.y += pod2.speed.y * t; 
                        Rebound(pod1, pod2);*/

                        //}
                        stop = true;
                    }

                    double t11 = this->CollisionTime(pod1, runner);
                    double t12 = this->CollisionTime(pod1, chaser);
                    double t21 = this->CollisionTime(pod2, runner);
                    double t22 = this->CollisionTime(pod2, chaser);

                
                    if(i == 0){

                        //if(isleader ||(!isleader && i <= 2)){
                        if(t11<= 1.0 ||t12 <=1.0){
                            solret.moves1[0].shield = 1;
                        }
                        else{
                            solret.moves1[0].shield = 0;
                        }
                        //}

                        //if(!isleader ||(isleader && i <= 2)){
                        if(t21<= 1.0 ||t22 <=1.0){
                            solret.moves2[0].shield = 1;
                        }
                        else{
                            solret.moves2[0].shield = 0;
                        }
                        //}
                    
                    }

                    
                    EndSimulate(pod1);
                    EndSimulate(pod2);
                    

                    int p1dist = distance(pod1.pos, this->checkpoints[pod1.check_point]);
                    int p2dist = distance(pod2.pos, this->checkpoints[pod2.check_point]);   
                
                    //if(isleader ||(!isleader && i <= 2)){
                    if(p1dist <= 600){
                        pod1.check_pass++;
                        pod1.check_point = (pod1.check_point + 1) % this->checkpoints.size();
                        
                    }
                // }

                    //if(!isleader ||(isleader && i <= 2)){
                    if(p2dist <= 600){
                        pod2.check_pass++;
                        pod2.check_point = (pod2.check_point + 1) % this->checkpoints.size();
                        
                    }
                // }


                }

                /*Vector2 checkp1=this->checkpoints[pod1.check_point], checkp2=this->checkpoints[pod2.check_point];
                    double g = dgene(rng);
                    checkp1.x += cos(g * 2.0 * PI) * 600.0;
                    checkp1.y += sin(g * 2.0 * PI) * 600.0;

                    g = dgene(rng);
                    checkp2.x += cos(g * 2.0 * PI) * 600.0;
                    checkp2.y += sin(g * 2.0 * PI) * 600.0;*/

                
                //scorep1 += distp1;
                //scorep2 += distp2;

                
                int p1dist = distance(pod1.pos, this->checkpoints[pod1.check_point]);
                int p2dist = distance(pod2.pos, this->checkpoints[pod2.check_point]);   
                
                scorep1 +=50000 * pod1.check_pass - p1dist;
                scorep2 += 50000 * pod2.check_pass - p2dist;
                
                /*if(scorep1 >= scorep2){
                    solret.score = scorep1;
                    solret.score += score_chaser;

                    state_chaser2 = 0;
                    
                }
                else{
                    solret.score = scorep2;
                    solret.score += score_chaser;

                    state_chaser = 0;
                }*/
                

                //solret.score += solret.moves1[0].thrust + solret.moves2[0].thrust;
                solret.score = scorep1+scorep2+solret.moves1[0].thrust + solret.moves2[0].thrust;
                

                if(solret.score > solution[NB_SOL-1].score){
                //cerr << solution[ind].score << endl;
                    solution[NB_SOL-1] = solret;
                    this->Trier();
                    
                }
                

                nb_turn++;
                
            
            }
            //cerr << "ned boucvle " << endl;



        }

        cerr << "turn "  << nb_turn << endl;

        Solutionm solm = solution[0];

        float anglef = int(p.angletot + solm.moves1[0].angle + 360) % 360;
        //cerr << "angle " << sim.angletot << endl;

        float angleRad = anglef * PI / 180.f;
        Vector2 dir = { cos(angleRad) * 10000.0f, sin(angleRad) * 10000.0f };
        int x = p.x + dir.x;
        int y = p.y + dir.y;

        int thrust = 0;
        thrust = solm.moves1[0].thrust ;
        int  thrust2 = 0;
        thrust2 = solm.moves2[0].thrust;

     
        anglef = int(p2.angletot + solm.moves2[0].angle + 360) % 360;
        //cerr << "angle " << sim.angletot << endl;

        angleRad = anglef * PI / 180.f;
        dir = { cos(angleRad) * 10000.0f, sin(angleRad) * 10000.0f };
        int x2 = p2.x + dir.x;
        int y2 = p2.y + dir.y;

        vector<string> ans;    
        if(solm.moves1[0].shield == 1)
            ans.push_back(to_string(x) + " " + to_string(y) + " SHIELD");
        else 
            ans.push_back(to_string(x) + " " + to_string(y) + " " + to_string(thrust));
    
        if(solm.moves2[0].shield == 1)
            ans.push_back(to_string(x) + " " + to_string(y) + " SHIELD");
        else
            ans.push_back(to_string(x2) + " " + to_string(y2) + " " + to_string(thrust2));
    
        return ans;

    }

    void Rebound(Sim& a, Sim& b)
    {
        // https://en.wikipedia.org/wiki/Elastic_collision#Two-dimensional_collision_with_two_moving_objects
        const float mA = 1;//Mass(a);
        const float mB = 1;//Mass(b);

        const Vector2 dP{b.pos.x - a.pos.x, b.pos.y - a.pos.y};
        const float AB = distance(a.pos, b.pos);

        const Vector2 u {dP.x * (1.f / AB),dP.y * (1.f / AB)}; // rebound direction

        const Vector2 dS = {b.speed.x - a.speed.x, b.speed.y - a.speed.y};

        const float m = (mA * mB) / (mA + mB);
        const float k = dot(dS, u);

        const float impulse = -2.f * m * k;
        const float impulseToUse = clamp(impulse, -minImpulse, minImpulse);

        a.speed.x += (-1.f/mA) * impulseToUse * u.x;
        a.speed.y += (-1.f/mA) * impulseToUse * u.y;
        b.speed.x += (1.f/mB) * impulseToUse * u.x;
        b.speed.y += (1.f/mB) * impulseToUse * u.y;
    }

    float CollisionTime(Sim& p1, Sim& p2)
    {
        const Vector2 dP{p2.pos.x - p1.pos.x, p2.pos.y - p1.pos.y};
        const Vector2 dS{p2.speed.x - p1.speed.x, p2.speed.y - p1.speed.y};

        constexpr float eps = 0.000001f; // float precision...

        // we're looking for t such that:
        // |            p2(t)           -           p1(t)           | < 2*podRadius
        // |(p2.position + t*p2.speed)  - (p1.position + t*p1.speed)| < 2*podRadius
        // |(p2.position - p1.position) -  t*(p2.speed - p1.speed)  | < 2*podRadius
        // |         dP                 +            t*dS           | < 2*podRadius
        // t^2 dS^2 + t 2dPdS + dP^2 < 4 podRadius^2
        // t^2  a   + t   b   +   c      = 0;

        const float a = dot(dS, dS);
        if (a < eps) // moving away from each other
            return INFINITY;

        const float b = -2.f*dot(dP, dS);
        const float c = dot(dP,dP) - 4.f*podRadiusSqr;

        const float delta = b*b - 4.f*a*c;
        if (delta < 0.f) // no solution
            return INFINITY;

        const float t = (b - sqrt(delta)) / (2.f * a);
        if (t <= eps)
            return INFINITY;

        return t;
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
        checkpoint.push_back({(float)checkpoint_x, (float)checkpoint_y});
    }


    /*MadPodracing pod1{100, 100, 4}, pod2{100, 100, 4};
    pod1.checkpoints = checkpoint;
    pod2.checkpoints = checkpoint;*/

    //_5G g5 = _5G(10, 10, 6);
    //g5.checkpoints = checkpoint;

    Simulation simul = Simulation(3,  7);
    simul.checkpoints = checkpoint;
    //InitNN neural = InitNN();
     

    Player p1, p2, _p1, _p2, _p3, _p4;// = fuckup.Play(10353, 986, 10, 10, 0, 1, 500);
    _p1.act_check = 1;
    _p2.act_check = 1;
    _p1.check_pass++;
    _p2.check_pass++;
    _p3.act_check = 1;
    _p4.act_check = 1;
    _p3.check_pass++;
    _p4.check_pass++;
    int check_stay;
    bool lock_stay = false;
    int pl_stay = 0;
    bool start = true;
    int turn = 0;
    // game loop
    while (1) {
        for (int i = 0; i < 2; i++) {
            int x; // x position of your pod
            int y; // y position of your pod
            int vx; // x speed of your pod
            int vy; // y speed of your pod
            int angle; // angle of your pod
            int next_check_point_id; // next check point id of your pod
          
            cin >> x >> y >> vx >> vy >> angle >> next_check_point_id; cin.ignore();
            
            if(i == 0){
                _p1.x = x;
                _p1.y = y;
                _p1.speed.x = vx;
                _p1.speed.y = vy;
                _p1.angletot = angle;
                _p1.check_point = next_check_point_id;
                cerr << angle << " sp " << norme1(_p1.speed) <<  endl;

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
                cerr << angle << " sp " << norme1(_p2.speed) <<  endl;

                if(next_check_point_id != _p2.act_check){
                    _p2.act_check = next_check_point_id;
                    _p2.check_pass++;
                   
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
                cerr << angle << " sp " << norme1(_p3.speed) <<  endl;
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
                cerr << angle << " sp " << norme1(_p4.speed) <<  endl;
                if(next_check_point_id != _p4.act_check){
                    _p4.act_check = next_check_point_id;
                    _p4.check_pass++;
                }
            }


        }

        //string ans_p1 = pod1.Play(_p1, _p2,_p3, _p4,  turn, 30);
        /*string ans_p1 = neural.Play_sim(_p1, checkpoint);
        string ans_p2 = pod2.Play(_p2,_p1,_p3, _p4, turn, 30);
       
        if(start){
            cout << to_string((int)checkpoint[_p1.check_point].x) + " " + to_string((int)checkpoint[_p1.check_point].y) + " BOOST" << endl;
            cout << to_string((int)checkpoint[_p2.check_point].x) + " " + to_string((int)checkpoint[_p2.check_point].y) + " BOOST" << endl;
            //cout << ans_p2 << endl;
        
        }
        else{
            cout << ans_p1 << endl;
            cout << ans_p2 << endl;
        }*/
        
        /*int maxscore = -2000000000;
        int indi = -1;
        Sim pod ;
        pod.angletot = _p1.angletot;
        pod.speed.x = _p1.speed.x;
        pod.speed.y = _p1.speed.y;
        pod.direction = { 0,0 };
        pod.pos = { (float)_p1.x, (float)_p1.y };
        pod.score = -1000000000;
        pod.check_point = _p1.check_point;
        pod.check_pass = _p1.check_pass;
        for(int i = 0;i < 7;++i){
            if( i == 6){
                pod.thrust = 50;
                pod.angle = -18;
            }
            if( i == 5){
                pod.thrust = 100;
                pod.angle = -13;
            }
            if( i == 4){
                pod.thrust = 150;
                pod.angle = -8;
            }
            if( i == 3){
                pod.thrust = 200;
                pod.angle = 0;
            }
            if( i == 2){
                pod.thrust = 150;
                pod.angle = 8;
            }
            if( i == 1){
                pod.thrust = 100;
                pod.angle = 13;
            }
            if( i == 0){
                pod.thrust = 50;
                pod.angle = 18;
            }

            Sim _pod = pod;
            double pt_nuis = 0;
            int score = g5.MiniMax(_pod , 4, -2000000000, 2000000000, false, 0, pt_nuis);
            if(score > maxscore){
                maxscore = score;
                indi = i;

            }

        }

        if( indi == 6){
            pod.thrust = 50;
            pod.angle = -18;
        }
        if( indi == 5){
            pod.thrust = 100;
            pod.angle = -13;
        }
        if( indi == 4){
            pod.thrust = 150;
            pod.angle = -8;
        }
        if( indi == 3){
            pod.thrust = 200;
            pod.angle = 0;
        }
        if( indi == 2){
            pod.thrust = 150;
            pod.angle = 8;
        }
        if( indi == 1){
            pod.thrust = 100;
            pod.angle = 13;
        }
        if( indi == 0){
            pod.thrust = 50;
            pod.angle = 18;
        }

        float anglef = int(pod.angletot + pod.angle + 360) % 360;
            //cerr << "angle " << sim.angletot << endl;

            float angleRad = anglef * PI / 180.f;
            Vector2 dir = { cos(angleRad) * (float)10000, sin(angleRad) * (float)10000 };
        int x = _p1.x + dir.x;
        int y = _p1.y + dir.y;
        cout << to_string(x)<< " " <<  to_string(y) << " "  <<  to_string(pod.thrust) << endl;
        cout << "0 0 200" << endl;*/

        if(start){
            cout << to_string((int)checkpoint[_p1.check_point].x) + " " + to_string((int)checkpoint[_p1.check_point].y) + " BOOST" << endl;
            cout << to_string((int)checkpoint[_p2.check_point].x) + " " + to_string((int)checkpoint[_p2.check_point].y) + " BOOST" << endl;
            //cout << ans_p2 << endl;
        
        }
        else{

            vector<string> ans = simul.Play(_p1, _p2,_p3, _p4, turn, 70);
            for(int i = 0;i< 2;++i){
                
                cout << ans[i] << endl;
                
            }

        }

        start = false;
        ++turn;
     
    }
}