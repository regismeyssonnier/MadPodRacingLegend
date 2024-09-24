#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <math.h>
#include <set>
#include <random>
#include <chrono>
#include <deque>
#include <cstdlib>
#include <ctime>
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
    ll score;
    int decompte = 0;
    int touch = false;
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
    int decompte = 0;
    int touch = false;
    int opp_touch=0;
    
    
};

class Node{
public:
    vector<Node*> child;
    Node *parent=nullptr;
    double par_score = 0;
    double ucb=0;
    double n=0;
    double w=0;
    int num = 0;
    ll score;
    bool terminal = false;
    string ans;
    string dir;
    Sim player;
    Sim playert2;
    Sim playert3;
    Sim player1;
    Sim player2;
    Sim player3;
    int choose_son;
    int depth=0;
    bool expand = false;
    long double variance=  0.0;
    long double mean = 0.0;
    string path;
    double highest = -500000000;
    double lowest = 500000000;

    Node(){};
};

class NeuralNetwork {
public:
    double LR;
    int size_input;
    int size_hidden;
    int size_output;

    vector<double> input, hidden, output;
    vector<double> hidden_b, output_b;
    vector<vector<double>> hidden_w, output_w;
    vector<double> cost;

    vector<vector<vector<double>>> network_w, bnetwork_w;
    vector<vector<double>> network, network_b, bnetwork_b;
    vector<double> etiquette;
    NeuralNetwork() {}
    NeuralNetwork(int sz_in, int sz_hid, int sz_output, double lr) :size_input(sz_in),
        size_hidden(sz_hid), size_output(sz_output), LR(lr) {

        this->input.resize(this->size_input, 0.0);
        this->hidden.resize(this->size_hidden, 0.0);
        this->output.resize(this->size_output, 0.0);
        this->cost.resize(this->size_output, 0.0);

        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<double> dhidden(0.01, 0.2);

        for (int i = 0; i < size_input; ++i) {
            hidden_w.push_back({});
            for (int j = 0; j < size_hidden; ++j) {
                hidden_w[i].push_back(dhidden(rng));
            }
        }

        for (int i = 0; i < size_hidden; ++i) {
            output_w.push_back({});
            for (int j = 0; j < size_output; ++j) {
                output_w[i].push_back(dhidden(rng));
            }
        }

        for (int i = 0; i < size_hidden; ++i) {
            hidden_b.push_back(dhidden(rng));
        }

        for (int i = 0; i < size_output; ++i) {
            output_b.push_back(dhidden(rng));
        }

        for (int i = 0; i < size_output; ++i) {
            cout << output_b[i] << endl;
        }

    }

    NeuralNetwork(vector<int>dimension, double LR) {
        this->LR = LR;
        cerr << "init " << endl;

        for (int i = 0; i < dimension.size(); ++i) {
            vector<double> dim(dimension[i], 0.0);
            this->network.push_back(dim);

            if (i > 0)
                this->network_b.push_back(dim);

            if (i < dimension.size() - 1) {
                vector<vector<double>> dim2(dimension[i], vector<double>(dimension[i + 1], 0.0));
                this->network_w.push_back(dim2);
            }


        }
        cerr << "init 2" << endl;
        this->cost.resize(dimension[dimension.size() - 1], 0.0);

        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<double> dhidden(-0.5, 0.5);

        for (int i = 0; i < network_w.size(); ++i) {
            for (int j = 0; j < network_w[i].size(); ++j) {
                for (int k = 0; k < network_w[i][j].size(); ++k) {
                    network_w[i][j][k] = dhidden(rng);
                }

            }

        }
        cerr << "init 3" << endl;
        for (int i = 0; i < network_b.size(); ++i) {
            for (int j = 0; j < network_b[i].size(); ++j) {
                network_b[i][j] = dhidden(rng);
            }
        }
        cerr << "init 4" << endl;

    }

    void SetEtiquette(vector<double> et) {
        this->etiquette.swap(et);
    }

    void Set_InputNet(vector<double> inp) {
        this->network[0].swap(inp);
    }

    void normalizeInput(std::vector<double>& input) {
        // Calculate mean
        double mean = 0.0;
        for (double val : input) {
            mean += val;
        }
        mean /= input.size();

        // Calculate standard deviation
        double stddev = 0.0;
        for (double val : input) {
            stddev += (val - mean) * (val - mean);
        }
        stddev = sqrt(stddev / input.size());

        // Normalize input data
        for (double& val : input) {
            val = (val - mean) / stddev;
        }
    }

    void ForwardNN() {
                //cerr << "start f " << network.size() << endl;
        for (int i = 0; i < network.size() - 2; ++i) {
            //cerr << "i " << i << endl;
            for (int j = 0; j < network[i + 1].size(); ++j) {
                //cerr << "j " << j << " " << network[i + 1].size() << endl;
                double h = 0.0;
                for (int k = 0; k < network[i].size(); ++k) {
                    //cerr << "k " << k << " " << network[i].size() << endl;
                    h += network[i][k] * network_w[i][k][j];
                }
                //cerr << "endf1 " << endl;
                h += network_b[i][j];
                network[i + 1][j] = this->getTanh(h);
                //cerr << "endf2 " << endl;
            }

        }

        int ind = network.size() - 2;
        for (int j = 0; j < network[ind + 1].size(); ++j) {
            // cerr << "j " << j << " " << network[i + 1].size() << endl;
            double h = 0.0;
            for (int k = 0; k < network[ind].size(); ++k) {
                // cerr << "k " << k << " " << network[i].size() << endl;
                h += network[ind][k] * network_w[ind][k][j];
            }
            //cerr << "endf1 " << endl;
            h += network_b[ind][j];
            network[ind + 1][j] = h; //this->sigmoid(h);
            // cerr << "endf2 " << endl;
        }



        //cerr << "start f2" << endl;
        for (int i = 0; i < network[network.size() - 1].size(); ++i) {
            cost[i] += network[network.size() - 1][i] - this->etiquette[i];
        }
        //cerr << "start f3" << endl;

    }

    void Set_Input(vector<double> inp) {
        input.swap(inp);
    }

    double getTanh(double v) {
        return tanh(v);
    }

    double getDTanh(double v) {
        return 1.0 - tanh(v) * tanh(v);
    }

    double dRelu(double v) {
        //return 1.0;
        if (v > 0.0)return 1.0;
        else return 0.0;
    }

    double Relu(double v) {
        //return 1.0;
        return max(0.0, v);
    }


    void Softmaxdep(const std::vector<double>& x, vector<double>& out) {
        std::vector<double> exp_x;
        double sum_exp_x = 0.0;

        // Calculer les exponentielles des éléments du vecteur
        for (const double& xi : x) {
            exp_x.push_back(std::exp(xi));
            sum_exp_x += std::exp(xi);
        }

        // Calculer le softmax
        int ind = 0;
        for (const double& exi : exp_x) {
            out[ind] = exi / sum_exp_x;
            ++ind;
        }


    }

    void Softmax(const std::vector<double>& x, std::vector<double>& out) {
        double max_x = *std::max_element(x.begin(), x.end()); // Trouver la valeur maximale de x

        double sum_exp_x = 0.0;

        // Calculer les exponentielles des éléments du vecteur
        for (const double& xi : x) {
            sum_exp_x += std::exp(xi - max_x); // Soustraire max_x
        }

        // Calculer le softmax
        for (size_t i = 0; i < x.size(); i++) {
            out[i] = std::exp(x[i] - max_x) / sum_exp_x; // Soustraire max_x
        }
    }

         
    void InitByDump(vector<vector<double>> hw,
        vector<vector<double>> ow,
        vector<double> ob,
        vector<double> hb
    ) {

        this->hidden_w.swap(hw);
        this->output_w.swap(ow);
        this->output_b.swap(ob);
        this->hidden_b.swap(hb);

    }

    void InitByDumpNN(vector<vector<vector<double>>> nw, vector<vector<double>> nb) {


        network_w.swap(nw);
        network_b.swap(nb);

    }

    
   

};




class InitNN {
public:
    vector<Vector2>checkpoints;
    NeuralNetwork nn;
    double angle18 = 0.0;
    InitNN() {

        vector<int> dimension = { 8, 128, 6 };
        nn = NeuralNetwork(dimension, 0.4);
        

    };

    double angleModulo360(double angle) {
        // Calcul du modulo 360
        double moduloAngle = fmod(angle, 360.0);
        // Si le résultat est négatif, ajoutez 360 pour le rendre positif
        if (moduloAngle < 0)
            moduloAngle += 360.0;
        return moduloAngle;
    }

    void Play(Player _p1 /*, double a1, double a2, double a3, double a4, double a5, double a6, double a7*/) {

        int nextcheck = (_p1.check_point + 1) % checkpoints.size();
        double x1 = _p1.x;
        double y1 = _p1.y;
        double x2 = checkpoints[_p1.check_point].x;
        double y2 = checkpoints[_p1.check_point].y;
        double x3 = checkpoints[nextcheck].x;
        double y3 = checkpoints[nextcheck].y;
        double angle = atan2(y1 - y2, x1 - x2) - atan2(y3 - y2, x3 - x2);
        angle = angle * 180.0 / PI;
        angle = fmod((angle + 180.0), 360.0);
        if (angle < 0.0)
            angle += 360.0;
        angle -= 180.0;
        double anglech = atan2(y2 - y1, x2 - x1);
        anglech = anglech * 180.0 / PI;
        // Ajustement par rapport à l'angle total (p.angletot)
        anglech = fmod(anglech - _p1.angletot + 540, 360) - 180;

        double col = double((_p1.speed.x * (checkpoints[_p1.check_point].x - _p1.x) + _p1.speed.y * (checkpoints[_p1.check_point].y - _p1.y))) /
            double(sqrt(_p1.speed.x * _p1.speed.x + _p1.speed.y * _p1.speed.y) *
                sqrt((checkpoints[_p1.check_point].x - _p1.x) * (checkpoints[_p1.check_point].x - _p1.x) + (checkpoints[_p1.check_point].y - _p1.y) * (checkpoints[_p1.check_point].y - _p1.y)) + 0.000001);
       
        double distcheck = distance(checkpoints[_p1.check_point], { (float)_p1.x, (float)_p1.y });

        double speed = norme1(_p1.speed);

        double a1 = (angle+180) / 360.0;
        double a2 = (anglech+180) / 360.0;
        double a3 = (col+1)/2.0;
        double a4 = (200000.0-distcheck) / 200000.0;
        double a5 = speed / 10000.0;
        double a6 = _p1.angletot / 360.0;
        double a7 = ((float)_p1.thrust/200.0);
        double a8 = (angle18 + 18.0) / 36.0;

       

        nn.Set_InputNet({ a1, a2, a3, a4, a5, a6, a7, a8 });
        //nn.normalizeInput(nn.network[0]);
        nn.SetEtiquette({ 0, 0 });
        nn.ForwardNN();

        double maxn = -100000.0;
        int indn = -1;
        for(int i = 0;i <6;++i){
            if(nn.network[nn.network.size()-1][i]> maxn){
                maxn = nn.network[nn.network.size()-1][i];
                indn = i;
            }
        }

        vector<vector<double>> sorties = {
            {-18.0, 0.0},   // sortie 1
            {0.0, 0.0},     // sortie 2
            {18.0, 0.0},    // sortie 3
            {-18.0, 200.0}, // sortie 4
            {0.0, 200.0},   // sortie 5
            {18.0, 200.0}   // sortie 6
        };

        double _angle = sorties[indn][0];
        double thrust = sorties[indn][1];
        angle18 = _angle;
        cerr << "angle " << _angle << endl;
        cerr << "thrust " << thrust << endl;


        float anglef = this->angleModulo360(_p1.angletot + _angle);
        //cerr << "angle " << sim.angletot << endl;

        float angleRad = anglef * PI / 180.f;
        Vector2 dir = { cos(angleRad) * 10000.0f, sin(angleRad) * 10000.0f };
        int x = /*checkpoints[_p1.check_point].x;//*/_p1.x + dir.x;
        int y = /*checkpoints[_p1.check_point].y;//*/_p1.y + dir.y;

        cout << x << " " << y << " " << (int)thrust << endl;



    };

    

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
    int MAX_TEMP;
    int ITER = 0;
    vector<vector<double>> sorties, sorties2 ;
    

    Simulation(int nbs, int d):NB_SOL(nbs), DEPTH(d){
        std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<int> dthrust(MINT, MAXT);
        std::uniform_int_distribution<int> dangle(MINA, MAXA);

        /*sorties = {
            {-18.0, 0.0},   // sortie 1
            {0.0, 0.0},     // sortie 2
            {18.0, 0.0},    // sortie 3
            {-18.0, 200.0}, // sortie 4
            {0.0, 200.0},   // sortie 5
            {18.0, 200.0}   // sortie 6
        };*/

        

       
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

    Simulation(int nbs, int d, int t):NB_SOL(nbs), DEPTH(d), MAX_TEMP(t){
        std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<int> dthrust(MINT, MAXT);
        std::uniform_int_distribution<int> dangle(MINA, MAXA);

        sorties = {
            {-18.0, 0.0},   // sortie 1
            {0.0, 0.0},     // sortie 2
            {18.0, 0.0},    // sortie 3
            {-18.0, 200.0}, // sortie 4
            {0.0, 200.0},   // sortie 5
            {18.0, 200.0}   // sortie 6
        };

        sorties2 = {
            {-18.0, 0.0},   // sortie 1
            {0.0, 0.0},     // sortie 2
            {18.0, 0.0},    // sortie 3
            {-18.0, 200.0}, // sortie 4
            {0.0, 200.0},   // sortie 5
            {18.0, 200.0}   // sortie 6
        };

        /*for(int a = -18.0;a <= 18.0;a+=6){
            for(int t = 0;t <= 200;t+=100){
                sorties.push_back({(double)a, (double)t});
            }
        }*/

        

       
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
        std::uniform_int_distribution<int> dsol(1, NB_SOL-1);
        std::uniform_real_distribution<long double> drand(0.0, 1.0);
     
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

        int nb_turn = 0;
        long double temp = this->MAX_TEMP, mine=200000000, maxe = -200000000;
        solution[1].score = -2000000000;
        long double best_score = 0, current_score = 0;
        
        while(getTime()){

            int score_chaser = 0;

            //for(int ind = 0;ind < NB_SOL;++ind){
                //int ind = dsol(rng);
                Solutionm solret = Mutate2(1, amplitude);
                
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
                    
                
               
                        pod1.angle = solret.moves1[i].angle;
                        pod1.thrust = solret.moves1[i].thrust;
                        Simulate(pod1);
                        pod1.angletot = int(pod1.angletot + pod1.angle + 360) % 360;
                        
                

             
                        pod2.angle = solret.moves2[i].angle;
                        pod2.thrust = solret.moves2[i].thrust;
                        Simulate(pod2);
                        pod2.angletot = int(pod2.angletot + pod2.angle + 360) % 360;
               

                    

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
                
                  
                    if(p1dist <= 600){
                        pod1.check_pass++;
                        pod1.check_point = (pod1.check_point + 1) % this->checkpoints.size();
                        
                    }
             

                  
                    if(p2dist <= 600){
                        pod2.check_pass++;
                        pod2.check_point = (pod2.check_point + 1) % this->checkpoints.size();
                        
                    }
            


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
                solret.score = /*scorep1+solret.moves1[0].thrust +*/ solret.moves2[0].thrust+scorep2;
                long double new_score = solret.score; ///(long double)solret.score / 10000.0;

                long double xp = exp(-((long double)new_score - (long double)solution[1].score) / temp);
                ///cerr << xp << endl;
                mine = min(mine, xp);
                maxe = max(maxe, xp);
                if(new_score > solution[1].score || drand(rng) < xp){
                    solution[1] = solret;
                    //current_score = new_score;
                    //this->Trier();
                }


                

                if(new_score > solution[0].score){
                    solution[0] = solret;
                    best_score = new_score;          
                }

                temp = 0.99 * temp;
                

                nb_turn++;
                
            
            //}
            //cerr << "ned boucvle " << endl;



        }

        

        Solutionm solm = solution[0];

        cerr << "turn "  << nb_turn << " " << solm.score << " " << temp << endl;
        cerr << mine << " " << maxe << endl; 

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

    int Selection(Node* root, Node** leaf, double scale){

        //std::mt19937 rng(std::random_device{}());
        //std::uniform_int_distribution<int> dexplor(1, 4);

        //long double explora = dexplor(rng);
        Node *node = root;
        int depth=0;
       
            
            for(int i = 0;i < node->child.size();++i){
                              
                //UCT
                if(node->child[i]->n != 0){
                    double ad = 0;
                    ad = sqrt(2.0*log(node->n) / node->child[i]->n);
                    node->child[i]->ucb = (long double)node->child[i]->score / (node->child[i]->n) + ad;
                    //cerr << node->child[i]->ucb << endl;
                }
                else{
                    node->child[i]->ucb = std::numeric_limits<long double>::max();
                }

                                              
                this->ITER++;

            }
            
            long double max_ucb= std::numeric_limits<double>::lowest();
            int ind = -1;
            for(int i = 0;i < node->child.size();++i){
            
                if(node->child[i]->ucb > max_ucb){
                    max_ucb = node->child[i]->ucb;
                    ind = i;
                }

                this->ITER++;
            }

    
                node = node->child[ind];
                depth++;
         

        *leaf = node;
        //cerr << "node="<< node << endl;
                
        return depth;


    }

    void Expand(Node *node, int depth){

        
        //direction
        for(int i = 0;i < 6;++i){
                        
            node->expand = true;
            Node *n = new Node();
            n->parent = node;
            n->depth = depth+1;
            n->num = i;
            this->ITER++;
            node->child.push_back(n);

        }

    }
    
  
    void Backpropagation(Node* node, double sc){

        // Backpropagation du score
        Node* par = node;
    
                        
        while(par != nullptr) {
            par->n++;  
            par->score += sc; 
            par->w++;
            this->ITER++;
            par = par->parent;
        }


    }

    void RSMCTS(Player &_p, Player &_p2,Player &_p3, Player &_p4, int turn, int time){

        //cerr << "enter " << endl;

        //std::mt19937 rng(std::random_device{}());
        //std::uniform_int_distribution<int> dplay(0, 5);
        //std::uniform_int_distribution<int> dg(0, 10);
  

        auto startm = high_resolution_clock::now();;
        int maxt = 0;
        auto getTime = [&]()-> bool {
            auto stop = high_resolution_clock::now();
            auto duration = duration_cast<milliseconds>(stop - startm);
            //cerr << duration.count() << endl;
            maxt = duration.count();
            return(maxt <= time);
        };
   

        this->ITER = 0;

        Node *root = new Node();
        root->parent = nullptr;
        root->score = 0;

        Node *root2 = new Node();
        root2->parent = nullptr;
        root2->score = 0;

        Node *root3 = new Node();
        root3->parent = nullptr;
        root3->score = 0;
             
        int num_dir=-1;
        int max_depth = -1000000;

        int DEPTH = 45;
        int nb_turn = 0;
        
        int NB_TIMES = 0;
        int MAX_TIMES = 10;

        Vector2 center{0, 0};
        for(int i = 0;i < this->checkpoints.size();++i){
            center.x += this->checkpoints[i].x;
            center.y += this->checkpoints[i].y;
        }
        center.x /= (float)this->checkpoints.size();
        center.y /= (float)this->checkpoints.size();


        double highest = -50000000;
        double lowest = 50000000;

        ll score_final = -2000000000;
        int indice_final = -1;
        int ind0 = -1;

        Sim pod3, pod4;
        pod3.angletot = _p3.angletot;
        pod3.speed.x = _p3.speed.x;
        pod3.speed.y = _p3.speed.y;
        pod3.direction = { 0,0 };
        pod3.pos = { (float)_p3.x, (float)_p3.y };
        pod3.score = -2000000000;
        pod3.check_point = _p3.check_point;
        pod3.check_pass = _p3.check_pass;

        pod4.angletot = _p4.angletot;
        pod4.speed.x = _p4.speed.x;
        pod4.speed.y = _p4.speed.y;
        pod4.direction = { 0,0 };
        pod4.pos = { (float)_p4.x, (float)_p4.y };
        pod4.score = -2000000000;
        pod4.check_point = _p4.check_point;
        pod4.check_pass = _p4.check_pass;

        Sim pod1, pod2;

        while(getTime() ){
            Node *_node_p1 =  root, *_node_p2 = root2, *_node_p3 = root3;
            Node *node_p1, *node_p2, *node_p3;
            highest = -500000000;
            lowest = 500000000;

                   
            ll scorep1 = 0.0, scorep2=0.0, scorep3 = 0.0;
            ll bb1 = 0.0;

            
 
            pod1.angletot = _p2.angletot;
            pod1.speed.x = _p2.speed.x;
            pod1.speed.y = _p2.speed.y;
            pod1.direction = { 0,0 };
            pod1.pos = { (float)_p2.x, (float)_p2.y };
            pod1.score = -2000000000;
            pod1.check_point = _p2.check_point;
            pod1.check_pass = _p2.check_pass;

            pod2.angletot = _p.angletot;
            pod2.speed.x = _p.speed.x;
            pod2.speed.y = _p.speed.y;
            pod2.direction = { 0,0 };
            pod2.pos = { (float)_p.x, (float)_p.y };
            pod2.score = -2000000000;
            pod2.check_point = _p.check_point;
            pod2.check_pass = _p.check_pass;
            pod2.touch = _p.touch;
            pod2.decompte = _p.decompte;
            pod2.opp_touch = 0;

            
           

       
            for(int depth = 0;getTime() && depth < 4;++depth){
                int d1=depth, d2=depth, d3=depth;
                if(_node_p1->child.size() == 0)
                    this->Expand(_node_p1, d1);
                if(_node_p2->child.size() == 0)
                    this->Expand(_node_p2, d2);
              
                max_depth = max(d1,max_depth);

                
                bool times = false;
              
                ll b1=-2000000000.0, b2 = -2000000000.0, b3 = -2000000000.0;
                int indm1=-1, indm2=-1, indm3 = -1;
                int m1 = -1, m2 = -1;
                /*int rd = rand() % 10;
                if(rd<1){
                    m1 = rand() % 6;
                    node_p1 = _node_p1->child[m1];
                    
                }
                else{*/
                    int depth_p1 = this->Selection(_node_p1, &node_p1, 1.0);
                    m1 = node_p1->num;
                //}

                /*if(rd < 1){
                    m2 = rand() % 6;
                    node_p2 = _node_p2->child[m2];
                   
                }
                else{*/
                    int depth_p2 = this->Selection(_node_p2, &node_p2, 1.0);
                    m2 = node_p2->num;
                //}

                NB_TIMES++;

                node_p1->n++;
                node_p2->n++;
               
                
                            
                pod1.angle = sorties[m1][0];
                pod1.thrust = sorties[m1][1];
                this->Simulate(pod1);
                pod1.angletot = int(pod1.angletot + pod1.angle +360) % 360;

                EndSimulate(pod1);
                int p1dist = distance(pod1.pos, this->checkpoints[pod1.check_point]);
                
                if(p1dist <= 600){
                    pod1.check_pass++;
                    pod1.check_point = (pod1.check_point + 1) % this->checkpoints.size();
                    
                }
                

                pod2.angle = sorties2[m2][0];
                pod2.thrust = sorties2[m2][1];
                this->Simulate(pod2);
                pod2.angletot = int(pod2.angletot + pod2.angle +360) % 360;

                EndSimulate(pod2);
                int p2dist = distance(pod2.pos, this->checkpoints[pod2.check_point]);
                
                if(p2dist <= 600){
                    pod2.check_pass++;
                    pod2.check_point = (pod2.check_point + 1) % this->checkpoints.size();
                    
                }

                double p3dist = distance(pod3.pos, this->checkpoints[_p3.check_point]);
                double score_int3 = 50000 *_p3.check_pass - p3dist;_p3.score = score_int3;

                double p4dist = distance(pod4.pos, this->checkpoints[_p4.check_point]);
                double score_int4 = 50000 *_p4.check_pass - p4dist;_p4.score = score_int4;

                double  dist2 = distance(pod2.pos, ((score_int3 >= score_int4)? pod3.pos : pod4.pos));
                if(dist2 < 800)pod2.opp_touch++;

            

                double t11 = this->CollisionTime(pod1, pod3);
                double t12 = this->CollisionTime(pod1, pod4);
                double t21 = this->CollisionTime(pod2, pod3);
                double t22 = this->CollisionTime(pod2, pod4);

                if(t11 <= 1.0 ){
                    Rebound(pod1, pod3);
                    
                }
                if(t12 <= 1.0){
                    Rebound(pod1, pod4);
                    
                }

                if(t21 <= 1.0 ){
                    Rebound(pod2, pod3);
                    
                }
                if(t22 <= 1.0 ){
                    Rebound(pod2, pod4);
                    
                }


                
                
                                                       
                       
                _node_p1 = node_p1;
                _node_p2 = node_p2;
       
            }

    
            double p1dist = distance(pod1.pos, this->checkpoints[pod1.check_point]);
            double score_int = 50000 * pod1.check_pass - p1dist;pod1.score = score_int;

            double col = this->Col(pod1, this->checkpoints[pod1.check_point]);


                        // Ajustement dynamique du threshold basé sur la déviation
            double threshold = 0.8;  // Le seuil où l'effet est maximal (0.5 radians)

            // Renforcer l'effet en dessous de 0.5 radians et l'atténuer au-dessus
            double weight;
            if (col < threshold) {
                weight = 10000.0;  // Renforcer l'effet quand angle_diff est inférieur à 0.5
            } else {
                weight = 0.0;  // Diminuer l'effet quand angle_diff est supérieur à 0.5
            }

                                  
            // Calcul du score final avec un ajustement plus fort autour de 0.5
            double score_finalr1 = score_int + col * weight;

          
            double p3dist = distance({(float)_p3.x, (float)_p3.y}, this->checkpoints[_p3.check_point]);
            double score_int3 = 50000 *_p3.check_pass - p3dist;_p3.score = score_int3;

            double p4dist = distance({(float)_p4.x, (float)_p4.y}, this->checkpoints[_p4.check_point]);
            double score_int4 = 50000 *_p4.check_pass - p4dist;_p4.score = score_int4;

            double  dist2 = distance(pod2.pos, ((score_int3 >= score_int4)? pod3.pos : pod4.pos));
          
            //double p2dist = distance(pod2.pos, this->checkpoints[pod2.check_point]);
            //double score_int2 = 50000 * pod2.check_pass- p2dist;pod2.score = score_int2;
            double score_int2 = exp(1.0-(dist2 / 50000.0)) * 10000;
            //double col2 = this->Col(pod3, pod2.pos);
            //double col22 = this->Col(pod2, pod3.pos);
            double colc = this->Col(pod2, center);

            // Ajustement dynamique du threshold basé sur la déviation
            double threshold2 = 0.0;  // Le seuil où l'effet est maximal (0.5 radians)

            // Renforcer l'effet en dessous de 0.5 radians et l'atténuer au-dessus
            double weight2;
            /*if (col2 < threshold2) {
                weight2 = 10000.0;  // Renforcer l'effet quand angle_diff est inférieur à 0.5
            } else {
                weight2 = 0.0;  // Diminuer l'effet quand angle_diff est supérieur à 0.5
            }*/

            double dist_center2 = distance(pod2.pos, center);

            /*if(dist_center2 < 1000 && col2 > 0.5 && !pod2.touch){
                pod2.decompte = 20;
                pod2.touch = true;
                //cerr << "touch" << endl;
            }
            else if(pod2.touch){
                pod2.decompte--;
                cerr << pod2.decompte << endl;
                if(pod2.decompte == 0){
                    pod2.touch = false;
                }
            }*/

            if(!pod2.touch){
                score_int2 = 50000 -dist_center2;
                weight2 = colc * 100000.0;
            }

            
            

            // Calcul du score final avec un ajustement plus fort autour de 0.5
            double score_finalr2 = score_int2 + weight2 + pod2.opp_touch * 100;

            

            //p2
            this->Backpropagation(node_p1, score_finalr1);
            //p1
            this->Backpropagation(node_p2, score_finalr2);
      
            nb_turn++;
          
       
        }

        int indc = 0;
        Node *node = root;
        //cerr << "calc" << endl;
        long double maxscore = std::numeric_limits<long double>::lowest();;
        //cerr << maxscore << " " << node->child.size() << endl;
        for(int i = 0;i < node->child.size();++i){
            long double score = 0.0;
            if(node->child[i]->n == 0)continue;
            score = node->child[i]->n;
            this->ITER++;
            //cerr << score <<  " -- " << node->child[i]->score << endl;
            if(score > maxscore ){
                maxscore = score;
                indc = i;
            
            }
        }

        int indc2 = 0;
        Node *node2 = root2;
        //cerr << "calc" << endl;
        long double maxscore2 = std::numeric_limits<long double>::lowest();;
        //cerr << maxscore2 << " " << node2->child.size() << endl;
        for(int i = 0;i < node2->child.size();++i){
            long double score = 0.0;
            if(node2->child[i]->n == 0)continue;
            score = node2->child[i]->n;
            this->ITER++;
            //cerr << score <<  " -- " << node2->child[i]->score << endl;
            if(score > maxscore2 ){
                maxscore2 = score;
                indc2 = i;
            
            }
        }


        Sim pd1, pd2; 
        pd2.angle = sorties[node->child[indc]->num][0];
        pd2.thrust = sorties[node->child[indc]->num][1];
        pd2.angletot = _p2.angletot;
        pd2.speed.x = _p2.speed.x;
        pd2.speed.y = _p2.speed.y;
        pd2.direction = { 0,0 };
        pd2.pos = { (float)_p2.x, (float)_p2.y };
        pd2.score = -2000000000;
        pd2.check_point = _p2.check_point;
        pd2.check_pass = _p2.check_pass;
        this->Simulate(pd2);
        EndSimulate(pd2);

        pd1.angle = sorties2[node2->child[indc2]->num][0];
        pd1.thrust = sorties2[node2->child[indc2]->num][1];
        pd1.angletot = _p.angletot;
        pd1.speed.x = _p.speed.x;
        pd1.speed.y = _p.speed.y;
        pd1.direction = { 0,0 };
        pd1.pos = { (float)_p.x, (float)_p.y };
        pd1.score = -2000000000;
        pd1.check_point = _p.check_point;
        pd1.check_pass = _p.check_pass;
        this->Simulate(pd1);
        EndSimulate(pd1);

        pod3.angletot = _p3.angletot;
        pod3.speed.x = _p3.speed.x;
        pod3.speed.y = _p3.speed.y;
        pod3.direction = { 0,0 };
        pod3.pos = { (float)_p3.x, (float)_p3.y };
        pod3.score = -2000000000;
        pod3.check_point = _p3.check_point;
        pod3.check_pass = _p3.check_pass;

        pod4.angletot = _p4.angletot;
        pod4.speed.x = _p4.speed.x;
        pod4.speed.y = _p4.speed.y;
        pod4.direction = { 0,0 };
        pod4.pos = { (float)_p4.x, (float)_p4.y };
        pod4.score = -2000000000;
        pod4.check_point = _p4.check_point;
        pod4.check_pass = _p4.check_pass;

        
        double t11 = this->CollisionTime(pd1, pod3);
        double t12 = this->CollisionTime(pd1, pod4);
        double t21 = this->CollisionTime(pd2, pod3);
        double t22 = this->CollisionTime(pd2, pod4);

        bool sh1 = false, sh2 = false;
        if(t11 <= 1.0 ){
            sh1 = true;
        }
        if(t12 <= 1.0){
            sh1 = true;
            
        }

        if(t21 <= 1.0 ){
            sh2 = true;
            
        }
        if(t22 <= 1.0 ){
            sh2 = true;
            
        }

        double p3dist = distance({(float)_p3.x, (float)_p3.y}, this->checkpoints[_p3.check_point]);
        double score_int3 = 50000 *_p3.check_pass - p3dist;_p3.score = score_int3;

        double p4dist = distance({(float)_p4.x, (float)_p4.y}, this->checkpoints[_p4.check_point]);
        double score_int4 = 50000 *_p4.check_pass - p4dist;_p4.score = score_int4;

        Sim poop = (score_int3 >= score_int4)? pod3 : pod4;

        double  dist2 = distance(pd1.pos, poop.pos);
        double score_int2 = exp(1.0-(dist2 / 50000.0)) * 10000;
        double col2 = this->Col(poop, pd1.pos);
        double col22 = this->Col(pd1, poop.pos);
        double dist_center2 = distance(pd1.pos, center);

        if(dist_center2 < 1000 && col2 > 0.5 && !_p.touch){
            _p.decompte = 100;
            _p.touch = true;
            //cerr << "touch" << endl;
        }
        else if(_p.touch){
            _p.decompte--;
            cerr << _p.decompte << endl;
            if(_p.decompte == 0){
                _p.touch = false;
            }
        }

       
        
        //this->ITER += 10000000;
       
        //cerr << "time " << maxt << " " << maxscore << "\n" << max_depth  << " " << nb_turn<< "\nITERATIONS=" << this->ITER << endl;
                
        float anglef = int(_p2.angletot + sorties[node->child[indc]->num][0] + 360) % 360;
        //cerr << "angle " << sim.angletot << endl;

        float angleRad = anglef * PI / 180.f;
        Vector2 dir = { cos(angleRad) * 10000.0f, sin(angleRad) * 10000.0f };
        int x = _p2.x + dir.x;
        int y = _p2.y + dir.y;

        int thrust = sorties[node->child[indc]->num][1];

        //p2
        anglef = int(_p.angletot + sorties2[node2->child[indc2]->num][0] + 360) % 360;
     
        angleRad = anglef * PI / 180.f;
        Vector2 dir2 = { cos(angleRad) * 10000.0f, sin(angleRad) * 10000.0f };
        int x2 = _p.x + dir2.x;
        int y2 = _p.y + dir2.y;

        int thrust2 = sorties2[node2->child[indc2]->num][1];
        if(col22 > 0.5)thrust2 = 200;
        
        cout << x2 << " " << y2 << " " << thrust2;
        if(sh1)cout << " SHIELD" << endl;
        else cout << endl;

        cout << x << " " << y << " " << thrust;
        if(sh2)cout << " SHIELD" << endl;
        else cout << endl;
        cout.flush();
  
    }

    double Col(Sim pod1, Vector2 pos2){

        return double((pod1.speed.x * (pos2.x - pod1.pos.x) + pod1.speed.y * (pos2.y - pod1.pos.y))) /
            double(sqrt(pod1.speed.x * pod1.speed.x + pod1.speed.y * pod1.speed.y) *
                sqrt((pos2.x - pod1.pos.x) * (pos2.x - pod1.pos.x) + (pos2.y - pod1.pos.y) * (pos2.y - pod1.pos.y)) + 0.000001);



    }



};

void OverrideAngle(Player& p, const Vector2& target)
{
    Vector2 dir = {target.x - p.x, target.y - p.y};
    const float norm = sqrt(dir.x*dir.x+dir.y*dir.y);
    dir.x /= norm;
    dir.y /= norm;
    float a = acos(dir.x) * 180.f / PI;
    if (dir.y < 0)
        a = (360.f - a);
    p.angletot = a;
}

int main()
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    srand(time(0));

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

    Simulation simul = Simulation(2, 7, 10000);
    simul.checkpoints = checkpoint;
    //InitNN neural = InitNN();

    InitNN nn = InitNN();
    nn.checkpoints = checkpoint;
     

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
                //cerr << angle << " sp " << norme1(_p1.speed) <<  endl;

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
                //cerr << angle << " sp " << norme1(_p2.speed) <<  endl;

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
                //cerr << angle << " sp " << norme1(_p3.speed) <<  endl;
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
                //cerr << angle << " sp " << norme1(_p4.speed) <<  endl;
                if(next_check_point_id != _p4.act_check){
                    _p4.act_check = next_check_point_id;
                    _p4.check_pass++;
                }
            }


        }

       
        if(start){
            //cout << to_string((int)checkpoint[_p1.check_point].x) + " " + to_string((int)checkpoint[_p1.check_point].y) + " BOOST" << endl;
            //cout << to_string((int)checkpoint[_p2.check_point].x) + " " + to_string((int)checkpoint[_p2.check_point].y) + " BOOST" << endl;
            OverrideAngle(_p1, checkpoint[_p1.check_point]);
            OverrideAngle(_p2, checkpoint[_p2.check_point]);
            
            simul.RSMCTS(_p1, _p2,_p3, _p4, turn, 998);
            //cout << ans_p2 << endl;
        
        }
        else{

            //vector<string> ans = simul.Play(_p1, _p2,_p3, _p4, turn, 70);
            //for(int i = 0;i< 2;++i){
                //nn.Play(_p1);
                simul.RSMCTS(_p1, _p2,_p3, _p4, turn, 73);
                
            //}

        }

        start = false;
        ++turn;
     
    }
}