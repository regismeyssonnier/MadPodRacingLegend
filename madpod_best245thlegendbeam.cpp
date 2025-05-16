#pragma GCC optimize("O3","unroll-loops","omit-frame-pointer","inline") //Optimization flags
#pragma GCC option("arch=native","tune=native","no-zero-upper") //Enable AVX
#pragma GCC target("avx")  //Enable AVX
#include <x86intrin.h> //AVX/SSE Extensions
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <math.h>
#include <set>
#include <random>
#include <chrono>
#include <deque>
#include <map>
#include <time.h>
#include <cstdlib>
#include <memory>
using namespace std::chrono;
using namespace std;

#define ll long long int

// -------------------------------------------------------------------
// magic numbers
constexpr int SIMULATIONTURNS = 4; // number of turns to simulate per solution
constexpr int SOLUTIONCOUNT = 6; // size of the population (genetic evolution)
// note: there are more magic numbers that make more sense in their specific scopes, just look for constexpr
// -------------------------------------------------------------------
// simulation constants
constexpr int maxThrust = 200;
constexpr int boostThrust = 650;
constexpr int maxRotation = 18;

constexpr int shieldCooldown = 4; // shield turn + 3

constexpr float checkpointRadius = 600.f;
constexpr float checkpointRadiusSqr = checkpointRadius * checkpointRadius;

constexpr float podRadius = 400.f;
constexpr float podRadiusSqr = podRadius * podRadius;

constexpr float minImpulse = 120.f;
constexpr float frictionFactor = .85f;

constexpr int timeoutFirstTurn = 500; //ms
constexpr int timeout = 75; //ms

constexpr float PI = 3.14159f;

vector<vector<double>> sorties = {
    {-18.0, 0.0},   // sortie 1
    {0.0, 0.0},     // sortie 2
    {18.0, 0.0},    // sortie 3
    {-18.0, 200.0}, // sortie 4
    {0.0, 200.0},   // sortie 5
    {18.0, 200.0}   // sortie 6
};

// -------------------------------------------------------------------
// basic 2D vector to simplify notations
struct Vector2
{
    float x;
    float y;
    Vector2(float _x, float _y) : x(_x), y(_y) {}
    Vector2() : x(0.f), y(0.f) {}

    Vector2 normalized() const
    {
        const float norm = sqrt(x*x+y*y);
        return Vector2(x/norm, y/norm);
    }
};
bool operator==(const Vector2& a, const Vector2& b)
{
    return a.x == b.x && a.y == b.y;
}
Vector2 operator+(const Vector2& a, const Vector2& b)
{
    return Vector2(a.x+b.x, a.y+b.y);
}
Vector2& operator+=(Vector2& a, const Vector2& b) {
    a.x += b.x;
    a.y += b.y;
    return a;
}
Vector2 operator-(const Vector2& a, const Vector2& b)
{
    return Vector2(a.x-b.x, a.y-b.y);
}
Vector2 operator*(float k, const Vector2& a)
{
    return Vector2(k*a.x, k*a.y);
}
Vector2& operator*=(Vector2& a, float k) {
    a.x *= k;
    a.y *= k;
    return a;
}
ostream& operator<<(ostream& os, const Vector2& v)
{
    os << v.x << " " << v.y;
    return os;
}
float distSqr(const Vector2& a, const Vector2& b)
{
    return (a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y);
}
float dist(const Vector2& a, const Vector2& b)
{
    return sqrt(distSqr(a, b));
}
float dot(const Vector2& a, const Vector2& b)
{
    return a.x*b.x + a.y*b.y;
}
// -------------------------------------------------------------------

struct Pod
{
    //int id = -1; // debug only

    // state
    Vector2 position;
    Vector2 speed;
    int angle = -1;

    // race status
    int checkpointId = 0;
    int checkpointsPassed = 0;

    // boost/shield
    bool canBoost = true;
    int shieldCd = 0;

    ll score = 0;
};




void ManageShield(bool turnOn, Pod& p)
{
  if (turnOn)
    p.shieldCd = shieldCooldown;
  else if (p.shieldCd > 0)
    --p.shieldCd;
}

int Mass(const Pod& p)
{
    if (p.shieldCd == shieldCooldown)
        return 10;
    return 1;
}

float CollisionTime(Pod& p1, Pod& p2)
{
    const Vector2 dP = (p2.position - p1.position);
    const Vector2 dS = (p2.speed - p1.speed);

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

void Rebound(Pod& a, Pod& b)
{
    // https://en.wikipedia.org/wiki/Elastic_collision#Two-dimensional_collision_with_two_moving_objects
    const float mA = Mass(a);
    const float mB = Mass(b);

    const Vector2 dP = b.position - a.position;
    const float AB = dist(a.position, b.position);

    const Vector2 u = (1.f / AB) * dP; // rebound direction

    const Vector2 dS = b.speed - a.speed;

    const float m = (mA * mB) / (mA + mB);
    const float k = dot(dS, u);

    const float impulse = -2.f * m * k;
    const float impulseToUse = clamp(impulse, -minImpulse, minImpulse);

    a.speed += (-1.f/mA) * impulseToUse * u;
    b.speed += (1.f/mB) * impulseToUse * u;
}

/*void print(const Pod& p)
{
    //if (p.id != 1) return;

    cerr << "position: " << p.position << endl;
    cerr << "speed:    " << p.speed << endl;
    cerr << "angle:    " << p.angle << endl;
    cerr << "nextCpId: " << p.checkpointId;
    cerr << "  passed: " << p.checkpointsPassed << endl;
    cerr << "canBoost: " << p.canBoost;
    cerr << "  shieldCd: " << p.shieldCd << endl << endl;
}*/

void UpdatePodFromInput(Pod& p, int id)
{
    int x, y, vx, vy, angle, nextCheckPointId;
    cin >> x >> y >> vx >> vy >> angle >> nextCheckPointId; cin.ignore();

    //p.id = id;
    p.position = Vector2(x,y);
    p.speed = Vector2(vx, vy);
    p.angle = angle;
    if (p.checkpointId != nextCheckPointId)
        ++p.checkpointsPassed;
    p.checkpointId = nextCheckPointId;

    //print(p);
}

struct Move
{
    int rotation = 0; // -maxRotation, maxRotation
    int thrust = 0; // 0, maxThrust
    bool shield = false;
    bool boost = false;
};
// the moves of both pods during a turn
class Turn
{
    private:
        vector<Move> moves = vector<Move>(4);
    public:
        Move& operator[](size_t m)             { return moves[m]; } //0<=m<2
        const Move& operator[](size_t m) const { return moves[m]; } //0<=m<2
};

struct Node{

    vector<shared_ptr<Node>> child;

    Pod pod;
    vector<Pod> tpod;
    int num_pod;

    ll score=0;
    Turn turn;


};

// Comparateur personnalisé pour comparer les ensembles par ordre décroissant
struct CustomCompare {
    bool operator()(const shared_ptr<Node> lhs, const shared_ptr<Node> rhs) const {
        return lhs->score > rhs->score;
    }
};

// all the moves to perform the next SIMULATIONTURNS turns
class Solution
{
    private:
        vector<Turn> turns = vector<Turn>(SIMULATIONTURNS);
    public:
        Turn& operator[](size_t t)             { return turns[t]; } //0<=t<SIMULATIONTURNS
        const Turn& operator[](size_t t) const { return turns[t]; } //0<=t<SIMULATIONTURNS

        int score = -1;
};

/*void print(const Solution& s)
{
    for (int t=0; t<SIMULATIONTURNS; t++)
    {
        for (int i=0; i<2; i++)
        {
            const Move& m = s[t][i];
            cerr << m.rotation << "     " << m.thrust << endl;
        }
        cerr << endl;
    }
}*/

class Simulation
{
    private:
        vector<Vector2> checkpoints;
        int checkpointCount;
        int maxCheckpoints;

    public:
        int MaxCheckpoints() const { return maxCheckpoints; }
        const vector<Vector2>& Checkpoints() const { return checkpoints; }

        Vector2 InitCheckpointsFromInput()
        {
            int laps;
            {
                cin >> laps; cin.ignore();
                cin >> checkpointCount; cin.ignore();
            }

            checkpoints.reserve(checkpointCount);
            for (int i=0; i<checkpointCount; i++)
            {
                int checkpointX, checkpointY;
                cin >> checkpointX >> checkpointY; cin.ignore();
                checkpoints[i] = Vector2(checkpointX, checkpointY);
            }

            maxCheckpoints = laps*checkpointCount;

            // return the first checkpoint (that the pods will face from the start)
            return checkpoints[1]; // checkpoints.size() > 1
        }

        void PlaySolution(vector<Pod>& pods, const Solution& s) const
        {
            for (int i=0; i<SIMULATIONTURNS; i++)
            {
                PlayOneTurn(pods, s[i]);
            }
        }

    private:
        void PlayOneTurn(vector<Pod>& pods, const Turn& turn) const
        {
            // those are the steps described in "expert rules", in order
            Rotate(pods, turn);
            Accelerate(pods, turn);
            UpdatePositions(pods);
            Friction(pods);
            EndTurn(pods);
        }

        void Rotate(vector<Pod>& pods, const Turn& turn) const
        {
            for (int i=0; i<4; i++)
            {
                Pod& p = pods[i];
                const Move& m = turn[i];

                p.angle = (p.angle + m.rotation) % 360;
            }
        }
        void Accelerate(vector<Pod>& pods, const Turn& turn) const
        {
            for (int i=0; i<4; i++)
            {
                Pod& p = pods[i];
                const Move& m = turn[i];

                // shield
                ManageShield(m.shield, p);
                if (p.shieldCd > 0)
                    continue; // no thrust this turn

                const float angleRad = p.angle * PI / 180.f;
                const Vector2 direction{cos(angleRad), sin(angleRad)};

                // boost
                const bool useBoost = p.canBoost && m.boost;
                const int thrust = useBoost ? boostThrust : m.thrust;
                if (useBoost)
                    p.canBoost = false;

                p.speed += thrust * direction;
            }
        }
        void UpdatePositions(vector<Pod>& pods) const
        {
            float t = 0.f;
            constexpr float turnEnd = 1.f;
            while (t < turnEnd)
            {
                // first pods to collide with each other
                Pod* a = nullptr;
                Pod* b = nullptr;
                float dt = turnEnd - t; // date of the collision (from t)

                for (int i=0; i<4; i++)
                {
                    for (int j=i+1; j<4; j++)
                    {
                        const float collisionTime = CollisionTime(pods[i], pods[j]);
                        if ( (t+collisionTime < turnEnd) && (collisionTime < dt) )
                        {
                            dt = collisionTime;
                            a = &pods[i];
                            b = &pods[j];
                        }
                    }
                }

                // move all the pods until the date we selected
                for (Pod& p : pods)
                {
                    p.position += dt * p.speed;

                    // collisions with checkpoints (do we need to be as accurate as pod collisions?)
                    if (distSqr(p.position, checkpoints[p.checkpointId]) < checkpointRadiusSqr)
                    {
                        p.checkpointId = (p.checkpointId + 1) % checkpointCount;
                        ++p.checkpointsPassed;
                    }
                }

                // a and b are colliding -> elastic collision
                if (a != nullptr && b != nullptr)
                {
                    //cerr << "collision at t=" << t << " + dt=" << dt;
                    //cerr << " between "<<a->id<<" and "<<b->id<<endl;
                    Rebound(*a, *b);
                }

                t+=dt;
            }
        }
        void Friction(vector<Pod>& pods) const
        {
            for (Pod& p : pods)
            {
                p.speed *= frictionFactor;
            }
        }
        void EndTurn(vector<Pod>& pods) const
        {
            for (Pod& p : pods)
            {
                p.speed = Vector2{ (int) p.speed.x, (int) p.speed.y };
                p.position = Vector2{round(p.position.x), round(p.position.y)};

                //print(p);
            }
        }

        int BestMove(vector<Pod> _pods, int num_pod, int runner){

            auto podScore = [&](const Pod& p) -> int
            {
                // the max distance between 2 checkpoints in the map (of size 16k*9k) is about 18k
                // we rate each checkpoint passed higher than that:
                constexpr int cpFactor = 30000;
                const int distToCp = dist(p.position, this->Checkpoints()[p.checkpointId]);
                return cpFactor*p.checkpointsPassed - distToCp;
            };

            ll score = -2000000000;
            int ind_s = -1;
            for(int i = 0;i < 6;++i){

                Move m;
                m.rotation = sorties[i][0];
                m.thrust = sorties[i][1];

                vector<Pod> pods = _pods;
               
                Pod& p = pods[num_pod];
                p.angle = (p.angle + m.rotation) % 360;

                const float angleRad = p.angle * PI / 180.f;
                const Vector2 direction{cos(angleRad), sin(angleRad)};

                // boost
                const bool useBoost = p.canBoost && m.boost;
                const int thrust = useBoost ? boostThrust : m.thrust;
                if (useBoost)
                    p.canBoost = false;

                p.speed += thrust * direction;
                p.position += 1.0 * p.speed;
                
                if (distSqr(p.position, checkpoints[p.checkpointId]) < checkpointRadiusSqr)
                {
                    p.checkpointId = (p.checkpointId + 1) % checkpointCount;
                    ++p.checkpointsPassed;
                }

                p.speed *= frictionFactor;

                p.speed = Vector2{ (int) p.speed.x, (int) p.speed.y };
                p.position = Vector2{round(p.position.x), round(p.position.y)};

        
                for (Pod& pd : pods){
                    pd.score = podScore(pd);
             
                }

                Pod myRacer;
                Pod myInterceptor ;
                Pod opponentRacer ;

                constexpr int aheadBias = 2; // being ahead is better than blocking the opponent
                ll sscore = 0;

                int aheadScore;
                int interceptorScore;
                if(num_pod < 2){

                    int racerIndex = -1;
                    if(runner == 1)racerIndex = num_pod;
                    else racerIndex = 1-num_pod;

                    myRacer = pods[racerIndex];
                    myInterceptor = pods[1-racerIndex];

                    opponentRacer = (pods[2].score > pods[3].score) ? pods[2] : pods[3];

                    aheadScore = (myRacer.score - opponentRacer.score);

                    Vector2 opponentCheckpoint;
                  
                    if(racerIndex != num_pod){
                        if(aheadScore > 0){
                            
                            int dist1 = dist(pods[2].position, myRacer.position);
                            int dist2 = dist(pods[3].position, myRacer.position);
                            opponentRacer = (dist1< dist2) ? pods[2] : pods[3];
                    
                            interceptorScore = -dist(myInterceptor.position,opponentRacer.position);// opponentCheckpoint);// +colint+colint2;
                        }
                        else{
                            opponentCheckpoint = this->Checkpoints()[opponentRacer.checkpointId];
                            interceptorScore = -dist(myInterceptor.position,opponentCheckpoint);
                        }
                        sscore =  50000 + interceptorScore;
                    }
                    else{
                        sscore = pods[num_pod].score;

                    }

                }
                else{
                    int racerIndex = (pods[2].score > pods[3].score) ? 2 : 3;
                    int interIndex = -1;
                    if(racerIndex == 2)interIndex = 3;
                    else interIndex = 2;

                    myRacer = pods[racerIndex];
                    myInterceptor = pods[interIndex];

                    opponentRacer = (pods[0].score > pods[1].score) ? pods[0] : pods[1];

                    aheadScore = (myRacer.score - opponentRacer.score);

                    Vector2 opponentCheckpoint = this->Checkpoints()[opponentRacer.checkpointId];
                    interceptorScore = -dist(myInterceptor.position,opponentCheckpoint);
                   
                    sscore =  aheadScore*aheadBias + interceptorScore;

                }

                /*if (myRacer.checkpointsPassed > this->MaxCheckpoints())
                    sscore = INFINITY; // I won

                else if (opponentRacer.checkpointsPassed > this->MaxCheckpoints())
                    sscore = -INFINITY; // opponent won
                else  */
                      
                    

                
                if(sscore > score){
                    score = sscore;
                    ind_s = i;
                }


            }

            return ind_s;


        }

    public:

        Turn BeamSearch(const vector<Pod> pods, int num_pod, int runner, int time){

            auto startm = high_resolution_clock::now();;
            int maxt = 0;
            auto getTime = [&]()-> bool {
                auto stop = high_resolution_clock::now();
                auto duration = duration_cast<milliseconds>(stop - startm);
                //cerr << duration.count() << endl;
                maxt = duration.count();
                return(maxt <= time);
            };

            auto podScore = [&](const Pod& p) -> int
            {
                // the max distance between 2 checkpoints in the map (of size 16k*9k) is about 18k
                // we rate each checkpoint passed higher than that:
                constexpr int cpFactor = 30000;
                const int distToCp = dist(p.position, this->Checkpoints()[p.checkpointId]);
                return cpFactor*p.checkpointsPassed - distToCp;
            };


            int MAX_WIDTH = 500;
            int depth = 0;

            shared_ptr<Node> root = make_shared<Node>();
            root->tpod = pods;
            root->num_pod = num_pod;

            /*int racerIndex = -1;
            if(runner == 1)racerIndex = num_pod;
            else racerIndex = 1-num_pod;

            int dist1 = dist(pods[2].position, pods[racerIndex].position);
            int dist2 = dist(pods[3].position, pods[racerIndex].position);
    */

            Node T0 = *root;
            ll scoref = -2000000000;

            multiset<shared_ptr<Node>, CustomCompare> beam;

            beam.insert(root);

            while (depth < 15 && getTime()) {

                //cerr << "startTime" << depth << endl;
    
                multiset<shared_ptr<Node>, CustomCompare> TF;
                multiset<shared_ptr<Node>, CustomCompare>::iterator itbeam = beam.begin();
    
                for (int W = 0; TF.size() < MAX_WIDTH && itbeam != beam.end(); ++W) {
                    shared_ptr<Node> t = *itbeam;
                    itbeam++;

                    vector<int> ind_s(4, 0);
                    for(int i = 0;i < 4;++i){
                        if (i == num_pod)continue;
                        ind_s[i] = this->BestMove(t->tpod, i, runner);
                    }

                    for(int i = 0;i < 6;++i){
                        shared_ptr<Node> song = make_shared<Node>();
                        song->tpod = t->tpod;
                        song->num_pod = t->num_pod;

                        Turn turn;
                        Move move;
                        move.rotation = sorties[i][0];
                        move.thrust = sorties[i][1];
                        turn[t->num_pod] = move;

                        

                        for(int j = 0;j < 4;++j){
                            if(j == t->num_pod)continue;
                            Move m;
                            m.rotation = sorties[ind_s[j]][0];
                            m.thrust = sorties[ind_s[j]][1];
                            turn[j] = m;
                        }

                        if(depth == 0)song->turn = turn;
                        else song->turn = t->turn;

                        Rotate(song->tpod, turn);
                        Accelerate(song->tpod, turn);
                        UpdatePositions(song->tpod);
                        Friction(song->tpod);
                        EndTurn(song->tpod);
                        
                        for (Pod& pd : song->tpod)
                            pd.score = podScore(pd);

                        int racerIndex = -1;
                        if(runner == 1)racerIndex = t->num_pod;
                        else racerIndex = 1-t->num_pod;

                        const Pod myRacer = song->tpod[racerIndex];
                        const Pod myInterceptor = song->tpod[1-racerIndex];

                        Pod opponentRacer = (song->tpod[2].score > song->tpod[3].score) ? song->tpod[2] : song->tpod[3];

                        //cerr << "score" << endl;
                        

                        constexpr int aheadBias = 2; // being ahead is better than blocking the opponent
                        ll sscore = 0;
                        // how far ahead are we?
                        const int aheadScore = (myRacer.score - opponentRacer.score);// + col * weight;

                        // can my interceptor block the opponent racer?
                        Vector2 opponentCheckpoint;
                        int interceptorScore;

                        if(racerIndex != num_pod){
                            if(aheadScore > 0){
                                int dist1 = dist(song->tpod[2].position, song->tpod[racerIndex].position);
                                int dist2 = dist(song->tpod[3].position, song->tpod[racerIndex].position);
                                opponentRacer = (dist1 < dist2) ? song->tpod[2] : song->tpod[3];
                    
                                interceptorScore = -dist(song->tpod[num_pod].position,opponentRacer.position);// opponentCheckpoint);// +colint+colint2;
                            }
                            else{
                                opponentCheckpoint = this->Checkpoints()[opponentRacer.checkpointId];
                                interceptorScore = -dist(song->tpod[num_pod].position,opponentCheckpoint);
                            }

                            sscore =  50000 + interceptorScore;
                        }
                        else{
                            sscore = song->tpod[num_pod].score;

                        }


                        
                        /*if (myRacer.checkpointsPassed > this->MaxCheckpoints())
                            sscore= INFINITY; // I won

                        else if (opponentRacer.checkpointsPassed > this->MaxCheckpoints())
                            sscore= -INFINITY; // opponent won
                        
                        else*/
                        

                        
                        song->score = t->score + sscore;

                        TF.insert(song);


                    }




                }

                beam.swap(TF);
                //cerr << "swap" << endl;
                if (!beam.empty()) {
                    shared_ptr<Node> b = *beam.begin();
                    if (b->score > scoref) {
                        T0 = *b;
                        scoref = b->score;
                    }
                }




                depth++;
            }

            cerr << "depth=" << depth << ", time=" << maxt << endl;

            return T0.turn;

        }






};

void ConvertSolutionToOutput(const Solution& sol, vector<Pod>& pods)
{
    constexpr int t = 0; // we must output the first turn of the solution

    for (int i=0; i<2; i++)
    {
        Pod& p = pods[i];
        const Move& m = sol[t][i];

        // generate coordinates from the rotation
        const float angle = (p.angle + m.rotation) % 360;
        const float angleRad = angle * PI / 180.f;

        constexpr float targetDistance = 10000.f; // compute the target arbitrarily far enough, to avoid rounding errors
        const Vector2 direction{ targetDistance*cos(angleRad),targetDistance*sin(angleRad) };
        const Vector2 target = p.position + direction;

        cout << round(target.x) << " " << round(target.y) << " ";

        Pod p3 = pods[2];
        Pod p4 = pods[3];

        double t11 = CollisionTime(p, p3);
        double t12 = CollisionTime(p, p4);

        bool shield = false;
        if (t11 <= 1.0 || t12 <= 1.0)shield = true;


        if (shield)
            cout << "SHIELD";
        else if (m.boost)
            cout << "BOOST";
        else
            cout << m.thrust;

        cout << " RM.inc" << endl;
    }
}

void ConvertSolutionToOutputB(const Turn &turn, vector<Pod>& pods)
{
    constexpr int t = 0; // we must output the first turn of the solution

    for (int i=0; i<2; i++)
    {
        Pod& p = pods[i];
        const Move& m = turn[i];

        // generate coordinates from the rotation
        const float angle = (p.angle + m.rotation) % 360;
        const float angleRad = angle * PI / 180.f;

        constexpr float targetDistance = 10000.f; // compute the target arbitrarily far enough, to avoid rounding errors
        const Vector2 direction{ targetDistance*cos(angleRad),targetDistance*sin(angleRad) };
        const Vector2 target = p.position + direction;

        cout << round(target.x) << " " << round(target.y) << " ";

        Pod p3 = pods[2];
        Pod p4 = pods[3];

        double t11 = CollisionTime(p, p3);
        double t12 = CollisionTime(p, p4);

        bool shield = false;
        if (t11 <= 1.0 || t12 <= 1.0)shield = true;


        if (shield)
            cout << "SHIELD";
        else if (m.boost)
            cout << "BOOST";
        else
            cout << m.thrust;

        cout << " RM.inc" << endl;
    }
}

void UpdatePodsShieldBoostForNextTurn(const Solution& sol, vector<Pod>& pods)
{
    // those values aren't read from the input, so we have to compute them
    for (int i=0; i<2; i++)
    {
        Pod& p = pods[i];
        const Move& m = sol[0][i];

        ManageShield(m.shield, p);

        if (p.shieldCd == 0 && m.boost)
            p.canBoost = false;
    }
}

// https://software.intel.com/en-us/articles/fast-random-number-generator-on-the-intel-pentiumr-4-processor
// apparently much faster than std::rand
inline int fastrand()
{
    static unsigned int g_seed = 100;
    g_seed = (214013*g_seed+2531011);
    return (g_seed>>16)&0x7FFF;
}
inline int rnd(int a, int b)
{
    return (fastrand() % (b-a)) + a;
}


class Solver
{
    private:
        vector<Solution> solutions;
        Simulation* sim;

    public:
        Solver(Simulation* parSimulation)
        {
            sim = parSimulation; // sim != nullptr
            InitPopulation();
            FirstTurnBoostHack();
        }

        const Solution& Solve(const vector<Pod>& pods, int time)
        {
            // we can keep iterating until we spend too much time
            using namespace std::chrono;
            auto start = high_resolution_clock::now();
            auto keepSolving = [&start, time]() -> bool
            {
                auto now = high_resolution_clock::now();
                auto d = duration_cast<milliseconds>(now - start);
                return (d.count() < time);
            };

            // init this turn
            for (int i=0; i<SOLUTIONCOUNT; ++i)
            {
                // re-use previous solutions (by shifting them by 1 turn)
                Solution& s = solutions[i];
                ShiftByOneTurn(s);
                ComputeScore(s, pods);
            }

            int step = 0;
            while (keepSolving())
            {
                // build and rate mutated versions of our solutions
                for (int i=0; i<SOLUTIONCOUNT; ++i)
                {
                    Solution& newSol = solutions[SOLUTIONCOUNT+i];
                    newSol = solutions[i];
                    Mutate(newSol);
                    ComputeScore(newSol, pods);
                }

                // sort the solutions by score (the worst ones will be replaced during the next step)
                std::sort( solutions.begin(), solutions.end(),
                           [](const Solution& a, const Solution& b) { return a.score > b.score; }
                         );
                ++step;
            }

            cerr << step << " evolution steps - best score: " << solutions[0].score << endl;
            return solutions[0];
        }

    private:
        void InitPopulation()
        {
            // 0 to (SOLUTIONCOUNT-1): actual solutions from the previous turn
            // SOLUTIONCOUNT to (2*SOLUTIONCOUNT-1): temporary solutions from Solve()
            solutions.resize(2*SOLUTIONCOUNT);

            // make starting solutions random
            for (int s=0; s<SOLUTIONCOUNT; s++)
                for (int t=0; t<SIMULATIONTURNS; t++)
                    for (int i=0; i<4; i++)
                        Randomize(solutions[s][t][i]);
        }

        void FirstTurnBoostHack()
        {
            // hack: just set the boost here if possible, don't waste time later since there's only 1 boost per race anyway...
            const float d = distSqr(sim->Checkpoints()[0], sim->Checkpoints()[1]); // size>1
            constexpr float boostDistanceThreshold = 5000.f;
            if (d < boostDistanceThreshold)
                return;

            //cerr << "Boosting the first turn" << endl;
            for (int i=0; i<4; i++)
            {
                for (int s=0; s<SOLUTIONCOUNT; ++s)
                    solutions[s][0][i].boost = true;
            }
        }

        void RandomizeO(Move& m, bool modifyAll = true) const
        {
            // if !modifyAll, we need to select which value to modify

            // TODO: find a cleaner way to rewrite this (but keep the probabilities compile time)...
            constexpr int all=-1, rotation=0, thrust=1, shield=2, boost=3;
            constexpr int         pr=5      , pt=pr+5,  ps=pt+1,  pb=ps+0; // note: (pb-ps)==0 because we're using FirstTurnBoostHack

            const int i = modifyAll ? -1 : rnd(0, pb);
            const int valueToModify = [i, modifyAll]() -> int
            {
                if (modifyAll)
                    return true;
                if (i<=pr)
                    return rotation;
                if (i<=pt)
                    return thrust;
                if (i<=ps)
                    return shield;
                return boost;
            }();
            auto modifyValue = [valueToModify](int parValue)
            {
                if (valueToModify == all)
                    return true;
                return valueToModify == parValue;
            };

            if (modifyValue(rotation))
            {
                // arbitrarily give more weight to -maxRotation, 0, maxRotation
                const int r = rnd(-2*maxRotation, 3*maxRotation);
                if (r > 2*maxRotation)
                    m.rotation = 0;
                else
                    m.rotation = clamp(r, -maxRotation, maxRotation);
            }
            if (modifyValue(thrust))
            {
                // arbitrarily give more weight to 0, maxThrust
                const int r = rnd(-0.5f * maxThrust, 2*maxThrust);
                m.thrust = clamp(r, 0, maxThrust);
            }
            if (modifyValue(shield))
            {
                if (!modifyAll || (rnd(0,10)>6)) // don't swap it for every init
                    m.shield = !m.shield;
            }
            if (modifyValue(boost))
            {
                if (!modifyAll || (rnd(0,10)>6)) // don't swap it for every init
                    m.boost = !m.boost;
            }
        }


        void Randomize(Move& m, bool modifyAll = true) const
        {
            

            if(modifyAll){
                //Rotation
                const int r = rnd(-2*maxRotation, 3*maxRotation);
                if (r > 2*maxRotation)
                    m.rotation = 0;
                else
                    m.rotation = clamp(r, -maxRotation, maxRotation);


                //Thrust
                const int rt = rnd(-0.5f * maxThrust, 2*maxThrust);
                m.thrust = clamp(rt, 0, maxThrust);
                
            }
            else{
                const int mute = rnd(0, 11);

                if(mute < 5){
                    //Rotation
                    const int r = rnd(-2*maxRotation, 3*maxRotation);
                    if (r > 2*maxRotation)
                        m.rotation = 0;
                    else
                        m.rotation = clamp(r, -maxRotation, maxRotation);
                }
                else if(mute <= 10){
                    //Thrust
                    const int rt = rnd(-0.5f * maxThrust, 2*maxThrust);
                    m.thrust = clamp(rt, 0, maxThrust);


                }
                else if(mute <= 11){
                    if (rnd(0,10)>6) // don't swap it for every init
                        m.shield = !m.shield;

                    if (rnd(0,10)>6) // don't swap it for every init
                        m.boost = !m.boost;
                }


            }


        }


        void ShiftByOneTurn(Solution& s) const
        {
            for (int t=1; t<SIMULATIONTURNS; t++)
                for (int i=0; i<4; i++)
                    s[t-1][i] = s[t][i];

            // create a new random turn at the end
            for (int i=0; i<4; i++)
            {
                Move& m = s[SIMULATIONTURNS-1][i];
                Randomize(m);
            }
        }

        void Mutate(Solution& s) const
        {
            // mutate one value for a random t,i
            int k = rnd(0, 2*SIMULATIONTURNS);
            Move& m = s[k/2][k%4];

            int h = rnd(0,1);
            Randomize(m, false);
        }

        int ComputeScore(Solution& sol, const vector<Pod>& pods) const
        {
            vector<Pod> podsCopy = pods;
            sim->PlaySolution(podsCopy, sol);
            sol.score = RateSolution(podsCopy);
            return sol.score;
        }
        int RateSolution(vector<Pod>& pods) const
        {
            auto podScore = [&](const Pod& p) -> int
            {
                // the max distance between 2 checkpoints in the map (of size 16k*9k) is about 18k
                // we rate each checkpoint passed higher than that:
                constexpr int cpFactor = 30000;
                const int distToCp = dist(p.position, sim->Checkpoints()[p.checkpointId]);
                return cpFactor*p.checkpointsPassed - distToCp;
            };
            for (Pod& p : pods)
                p.score = podScore(p);

            const int myRacerIndex = (pods[0].score > pods[1].score) ? 0 : 1;
            const Pod& myRacer = pods[myRacerIndex];
            const Pod& myInterceptor = pods[1-myRacerIndex];

            const Pod& opponentRacer = (pods[2].score > pods[3].score) ? pods[2] : pods[3];

            if (myRacer.checkpointsPassed > sim->MaxCheckpoints())
                return INFINITY; // I won

            if (opponentRacer.checkpointsPassed > sim->MaxCheckpoints())
                return -INFINITY; // opponent won

            /*double col = this->Col(myRacer,sim->Checkpoints()[myRacer.checkpointId]);

                       
            double threshold = 0.95;  

            double weight;
            if (col < threshold) {
                weight = 10000.0; 
            } else {
                weight = 0.0; 
            }*/

            // how far ahead are we?
            const int aheadScore = (myRacer.score - opponentRacer.score);// + col * weight;


            /*double col1 = this->Col(opponentRacer, myInterceptor.position);
            double col2 = this->Col(myInterceptor, opponentRacer.position);
            double colint = max(0.0, col1*10000);
            double colint2 = max(0.0, col2*10000);*/
        

            // can my interceptor block the opponent racer?
            const Vector2 opponentCheckpoint = sim->Checkpoints()[opponentRacer.checkpointId];
            const int interceptorScore = -dist(myInterceptor.position, opponentCheckpoint);// +colint+colint2;

            constexpr int aheadBias = 2; // being ahead is better than blocking the opponent
            return aheadScore*aheadBias + interceptorScore;
        }

        double Col(Pod pod1, Vector2 pos2) const{

            return double((pod1.speed.x * (pos2.x - pod1.position.x) + pod1.speed.y * (pos2.y - pod1.position.y))) /
                double(sqrt(pod1.speed.x * pod1.speed.x + pod1.speed.y * pod1.speed.y) *
                    sqrt((pos2.x - pod1.position.x) * (pos2.x - pod1.position.x) + (pos2.y - pod1.position.y) * (pos2.y - pod1.position.y)) + 0.000001);



        }

        





};

// make p face target immediately, ignoring turn rate (maxRotation): this is necessary to init the first turn
void OverrideAngle(Pod& p, const Vector2& target)
{
    const Vector2 dir = (target - p.position).normalized();
    float a = acos(dir.x) * 180.f / PI;
    if (dir.y < 0)
        a = (360.f - a);
    p.angle = a;
}


int main()
{
    Simulation simulation;
    const Vector2 firstCheckpoint = simulation.InitCheckpointsFromInput();

    Solver solver{ &simulation };
    vector<Pod> pods(4);

    auto podScore = [&](const Pod& p) -> int
    {
        // the max distance between 2 checkpoints in the map (of size 16k*9k) is about 18k
        // we rate each checkpoint passed higher than that:
        constexpr int cpFactor = 30000;
        const int distToCp = dist(p.position, simulation.Checkpoints()[p.checkpointId]);
        return cpFactor*p.checkpointsPassed - distToCp;
    };

    int step=0;
    while (1)
    {
        for (int i=0; i<4; i++)
        {
            UpdatePodFromInput(pods[i], i);
            if (step == 0)
                OverrideAngle(pods[i], firstCheckpoint);
        }

        const int availableTime = (step==0) ? timeoutFirstTurn : timeout;
        constexpr float timeoutSafeguard = .95f;

        int scorep1 = podScore(pods[0]);
        int scorep2 = podScore(pods[1]);

        int runner1 = 1;
        int runner2 = 0;
        if(scorep2 > scorep1){
            runner1 = 0;
            runner2 = 1;
        }

        //const Solution& s = solver.Solve(pods, availableTime*timeoutSafeguard);
        Turn t = simulation.BeamSearch(pods, 0, runner1, 25);
        Turn t2 = simulation.BeamSearch(pods, 1, runner2, 25);
        Turn tf;
        tf[0] = t[0];
        tf[1] = t2[1];
        ConvertSolutionToOutputB(tf, pods);

        //UpdatePodsShieldBoostForNextTurn(s, pods);

        ++step;
    }
}