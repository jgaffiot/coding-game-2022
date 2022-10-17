#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

using uint = unsigned int;

// some mathematical constants
constexpr double Pi() {
    return 3.14159265358979323846264338327950288419716939937510;
}  // 50 digits
constexpr double TwoPi() {
    return 2.0 * Pi();
}
constexpr double PiSq() {
    return Pi() * Pi();
}
constexpr double PiOver2() {
    return Pi() / 2.0;
}
constexpr double PiOver4() {
    return Pi() / 4.0;
}
constexpr double InvPi() {
    return 1.0 / Pi();
}
constexpr double RadToDeg() {
    return 180.0 / Pi();
}
constexpr double DegToRad() {
    return Pi() / 180.0;
}
constexpr double Sqrt2() {
    return 1.4142135623730950488016887242097;
}

// floating point comparison
template<
    typename Floating,
    typename = typename std::enable_if<std::is_floating_point<Floating>::value>>
bool equals(Floating A, Floating B) {
    // Check if the numbers are really close -- needed
    // when comparing numbers near zero.
    Floating diff = fabs(A - B);
    if (diff <= 2 * std::numeric_limits<Floating>::epsilon())
        return true;

    A = fabs(A);
    B = fabs(B);
    Floating largest = (B > A) ? B : A;

    if (diff <= largest * std::numeric_limits<Floating>::epsilon())
        return true;
    return false;
}

// Some mathematical functions with unlimited range thanks to variadic templates
template<typename T>
inline T sq(T val) {
    return val * val;
}
template<typename T>
inline T square(T val) {
    return val * val;
}
// NOLINTNEXTLINE(runtime/int)
inline unsigned long long int operator"" _sq(unsigned long long int val) {
    return val * val;
}
inline long double operator"" _sq(long double val) {
    return val * val;
}

template<typename T>
inline T cube(T val) {
    return val * val * val;
}
// NOLINTNEXTLINE(runtime/int)
inline unsigned long long int operator"" _cube(unsigned long long int val) {
    return val * val * val;
}
inline long double operator"" _cube(long double val) {
    return val * val * val;
}

/// power with integer exponent
template<typename T>
inline T pow_n(T base, unsigned int exp) {
    if (exp == 0u) {
        return static_cast<T>(1);
    }
    T result = static_cast<T>(1);
    while (exp) {
        if (exp & 1) {
            result *= base;
        }
        exp >>= 1;
        base *= base;
    }
    return result;
}

/// Sum of squares, result given in double
template<typename T>
inline double sum2(T val) {
    return static_cast<double>(val) * static_cast<double>(val);
}
template<typename T, typename... Args>
inline double sum2(T val, Args... args) {
    return static_cast<double>(val) * static_cast<double>(val) + sum2(args...);
}

/// Norm 2, the square root of the sum of squares, result given in double
template<typename... Args>
inline double norm2(Args... args) {
    return std::sqrt(sum2(args...));
}

/// Difference of square
template<typename T, typename U>
inline double diff2(T a, U b) {
    return static_cast<double>(a) * static_cast<double>(a)
           - static_cast<double>(b) * static_cast<double>(b);
}

namespace detail_string
{
template<typename S, typename T, typename = void>
struct is_streamable: std::false_type {};

template<typename S, typename T>
struct is_streamable<
    S,
    T,
    std::void_t<decltype(std::declval<S&>() << std::declval<T>())>>: std::true_type {};

template<
    typename T,
    typename = typename std::enable_if<is_streamable<std::ostringstream, T>{}>::type>
inline void PushToStream(std::ostringstream& oss, T&& val) {
    oss << std::forward<T>(val);
}

template<
    typename T,
    typename... Args,
    typename = typename std::enable_if<is_streamable<std::ostringstream, T>{}>::type>
inline void PushToStream(std::ostringstream& oss, T&& val, Args&&... args) {
    oss << std::forward<T>(val);
    PushToStream(oss, std::forward<Args>(args)...);
}
}  // end namespace detail_string

//! Concatenation of y_ streamable objects into a string, returns a std::string
template<typename... Args>
inline std::string cat(Args&&... args) {
    std::ostringstream oss;
    detail_string::PushToStream(oss, std::forward<Args>(args)...);
    return oss.str();
}

//! An operator<< on all enum types, enabling easy printing to log or debug
template<typename Enum>
inline std::ostream& operator<<(
    typename std::enable_if<std::is_enum<Enum>::value, std::ostream>::type& stream,
    const Enum& e) {
    return stream << static_cast<typename std::underlying_type<Enum>::type>(e);
}

//! The local exception
class Error: public std::exception {
public:
    virtual ~Error() noexcept {}

    template<typename... Args>
    // NOLINTNEXTLINE
    Error(Args... args): std::exception() {
        message = cat(args...);
    }

    const std::string& swhat() const noexcept { return message; }
    const char* what() const noexcept override { return message.c_str(); }

private:
    std::string message;
};

// Some constants
constexpr uint X_MIN{0u};
constexpr uint Y_MIN{0u};
constexpr uint X_MAX{17630u};
constexpr uint Y_MAX{9000u};
constexpr uint NB_HEROES{3u};
constexpr uint HERO_MOV{800u};  // hero maximal movement per turn
constexpr uint MSTR_MOV{400u};  // monster maximal movement per turn
constexpr uint DMG{2u};  // damages per turn
constexpr uint R_DMG{800u};  // damage radius
constexpr uint R_BASE{300u};  // base radius
constexpr uint R_FOG_BASE{6000u};  // base fog radius
constexpr uint R_FOG_HERO{2200u};  // hero fog radius
constexpr uint R_WIND{1280u};  // wind spell effect radius
constexpr uint WIND_MOV{2200u};  // wind spell movement
constexpr uint WIND_MANA{10u};

//! A 2-dimension mathematical vector to represent speed
class Vector2 {
public:
    Vector2() {}
    Vector2(double x, double y): v({x, y}) {}

    //! Create a Vector2 from polar coordinates.
    static Vector2 from_polar(double r, double theta) {
        return Vector2(abs(r) * cos(theta), abs(r) * sin(theta));
    }

    Vector2(const Vector2&) = default;
    Vector2& operator=(const Vector2&) = default;

    double operator[](std::size_t i) const { return v[i]; }
    double operator()(std::size_t i) const {
        if (i < 2) {
            return v[i];
        }
        throw Error("Vector2::operator(): index too high: ", i, " > 1");
    }
    double at(std::size_t i) const { return v.at(i); }
    double x() const { return v[0]; }
    double y() const { return v[1]; }

    Vector2& unit() {
        const double norm = norm2(v[0], v[1]);
        v[0] /= norm;
        v[1] /= norm;
        return *this;
    }
    double r() const { return norm2(v[0], v[1]); }  //! Modulus
    double theta() const { return atan2(v[0], v[1]); }  //! Argument

    void set_x(double x) { v[0] = x; }
    void set_y(double y) { v[1] = y; }
    void set(double x, double y) {
        v[0] = x;
        v[1] = y;
    }

    void set_r_theta(double r, double theta) {
        double amag = abs(r);
        v[0] = amag * cos(theta);
        v[1] = amag * sin(theta);
    }
    Vector2& rotate(double phi) {
        const double sinphi = sin(phi);
        const double cosphi = cos(phi);
        const double t = v[0] * cosphi - v[1] * sinphi;
        v[1] = v[1] * cosphi + v[0] * sinphi;
        v[0] = t;
        return *this;
    }
    friend Vector2 operator+(const Vector2& a, const Vector2& b) {
        return Vector2(a.x() + b.x(), a.y() + b.y());
    }
    friend Vector2 operator-(const Vector2& a, const Vector2& b) {
        return Vector2(a.x() - b.x(), a.y() - b.y());
    }
    friend double operator*(const Vector2& a, const Vector2& b) {
        return a.x() * b.x() + a.y() * b.y();
    }
    friend Vector2 operator*(const Vector2& v, double d) {
        return Vector2(d * v.x(), d * v.y());
    }
    friend Vector2 operator*(double d, const Vector2& v) {
        return Vector2(d * v.x(), d * v.y());
    }
    friend Vector2 operator/(const Vector2& v, double d) {
        return Vector2(v.x() / d, v.y() / d);
    }
    friend bool operator==(const Vector2& a, const Vector2& b) {
        return equals(a.x(), b.x()) and equals(a.y(), b.y());
    }
    friend std::ostream& operator<<(std::ostream& os, const Vector2& v) {
        os << "(" << v[0] << "," << v[1] << ")";
        return os;
    }

private:
    std::array<double, 2> v = {{}};
};

//! A 2 dimension point.
class Point {
    uint x_, y_;

public:
    Point(): x_(0u), y_(0u) {}
    Point(uint x, uint y): x_(x), y_(y) {
        if (x > X_MAX or y > Y_MAX) {
            throw Error(
                "Point out of bounds: ", x, " > ", X_MAX, " or ", y, " > ", Y_MAX);
        }
    }

    //! Create a Point from polar coordinates.
    static Point from_polar(double r, double theta) {
        return Point(abs(r) * cos(theta), abs(r) * sin(theta));
    }

    uint x() const { return x_; }
    uint y() const { return y_; }
    uint at(bool i) const { return i ? y_ : x_; }

    double r() const { return norm2(x_, y_); }  //! Modulus
    double theta() const { return atan2(x_, y_); }  //! Argument
    //! Distance to another point
    double dist_to(const Point& p) const {
        return norm2(
            x_ > p.x() ? x_ - p.x() : p.x() - x_, y_ > p.y() ? y_ - p.y() : p.y() - y_);
    }

    //! Advance the point from the given dist.
    void advance(int dist) {
        const double t = theta();
        x_ += dist * cos(t);
        y_ += dist * sin(t);
        x_ = min(max(0u, x_), X_MAX);
        y_ = min(max(0u, y_), Y_MAX);
    }
    void advance(const Vector2& speed) {
        x_ += speed.x();
        y_ += speed.y();
        x_ = min(max(0u, x_), X_MAX);
        y_ = min(max(0u, y_), Y_MAX);
    }
    void rotate(double phi) {
        const double sinphi = sin(phi);
        const double cosphi = cos(phi);
        const double t = x_ * cosphi - y_ * sinphi;
        y_ = y_ * cosphi + x_ * sinphi;
        x_ = t;
    }

    friend std::ostream& operator<<(std::ostream& os, const Point& p) {
        os << "Point(" << p.x() << ", " << p.y() << ")";
        return os;
    }

    bool operator==(const Point& other) { return x_ == other.x() and y_ == other.y(); }
    Point operator+(const Point& other) {
        return Point(x_ + other.x(), y_ + other.y());
    }
    Point operator+(const Vector2& speed) {
        return Point(x_ + speed.x(), y_ + speed.y());
    }
    Point operator-(const Point& other) {
        return Point(x_ - other.x(), y_ - other.y());
    }
    Point operator*(uint d) { return Point(d * x_, d * y_); }
};

enum class Type
{
    kMonster,
    kHero,
    kOpponent
};

// global variable to indicate if our base is top left (else bottom rifght)
static const array<Point, 2> kBases{Point{X_MAX, Y_MAX}, Point{X_MIN, Y_MIN}};
bool top_left{true};
int sign{0};
const Point& Home() {
    return kBases[top_left];
}
const Point& Opponent() {
    return kBases[not static_cast<bool>(top_left)];
}

//! An entity on the play field
class Entity {
public:
    Entity(uint id, const Point& p, Type type): id_(id), p_(p), type_(type) {}
    virtual ~Entity() {}
    Entity(const Entity&) = default;
    constexpr Entity(Entity&&) = default;
    Entity& operator=(const Entity&) = default;
    Entity& operator=(Entity&&) = default;

    uint id() const { return id_; }
    Point p() const { return p_; }
    Type type() const { return type_; }

    double dist_to(const Point& p) const { return p_.dist_to(p); }

    void update(uint x, uint y) { p_ = Point(x, y); }

    friend std::ostream& operator<<(std::ostream& os, const Entity& e) {
        os << "Entity(" << e.id() << ", " << e.p().x() << ", " << e.p().y() << ")";
        return os;
    }

private:
    uint id_;
    Point p_;
    Type type_;
};

enum class Threat
{
    kNone,
    kBase,
    kOpponent
};

class Monster: public Entity {
public:
    Monster(
        uint id,
        const Point& p,
        uint health,
        int vx,
        int vy,
        bool has_target,
        int threat):
        Entity(id, p, Type::kMonster),
        health_(health),
        v_(vx, vy),
        has_target_(has_target),
        threat_(Threat(threat)) {}
    Monster(
        int id, int x, int y, int health, int vx, int vy, int has_target, int threat):
        Entity(
            static_cast<uint>(id),
            {static_cast<uint>(x), static_cast<uint>(y)},
            Type::kMonster),
        health_(static_cast<uint>(health)),
        v_(vx, vy),
        has_target_(static_cast<bool>(has_target)),
        threat_(Threat(threat)) {}
    Monster(const Monster&) = default;
    constexpr Monster(Monster&&) = default;
    Monster& operator=(const Monster&) = default;
    Monster& operator=(Monster&&) = default;

    uint health() const { return health_; }
    Vector2 v() const { return v_; }
    bool has_target() const { return has_target_; }
    Threat threat() const { return threat_; }

    void update(
        uint x, uint y, uint health, int vx, int vy, bool has_target, int threat) {
        Entity::update(x, y);
        health_ = health;
        v_ = Vector2(vx, vy);
        has_target_ = has_target;
        threat_ = Threat(threat);
    }

    friend std::ostream& operator<<(std::ostream& os, const Monster& m) {
        os << "Monster(" << m.id() << ", " << m.p().x() << ", " << m.p().y() << ", "
           << m.health() << ", " << m.v() << ", "
           << (m.has_target() ? "targetting" : "wandering") << ", " << m.threat()
           << ")";
        return os;
    }

    static struct {
        bool operator()(const Monster& a, const Monster& b) {
            auto d_a = a.dist_to(Home());
            auto d_b = b.dist_to(Home());
            return d_a > d_b;
        }
    } cmp;

private:
    uint health_;
    Vector2 v_;
    bool has_target_;
    Threat threat_;
};

class Hero: public Entity {
public:
    Hero(uint id, const Point& p): Entity(id, p, Type::kHero) {}
    Hero(int id, int x, int y):
        Entity(
            static_cast<uint>(id),
            {static_cast<uint>(x), static_cast<uint>(y)},
            Type::kHero) {}

    void update(uint x, uint y) {
        prev_ = p();
        Entity::update(x, y);
        is_done_ = false;
    }

    void done() { is_done_ = true; }
    bool is_done() const { return is_done_; }

    Point Scout() {
        if (id() == 0) {
            return Standby();
        }

        double deviation = Pi() / 3.;
        if (id() == 2) {
            deviation *= -1.;
        }

        Point dest;
        double dist = dist_to(Home());
        if (dist > 15000) {
            dest = Home();
        } else if (dist > R_FOG_BASE + R_FOG_HERO) {
            dest = dest_;
        } else {
            dest = Opponent();
            dest.rotate(deviation);
        }
        dest_ = dest;
        return dest;
    }

    uint get_nb_target(const map<int, Monster>& monsters) {
        uint nb{0u};
        for (const auto& [id, monster] : monsters) {
            if (dist_to(monster.p()) < R_WIND) {
                nb++;
            }
        }
        return nb;
    }

    friend std::ostream& operator<<(std::ostream& os, const Hero& h) {
        os << "Hero(" << h.id() << ", " << h.p().x() << ", " << h.p().y() << ")";
        return os;
    }

    //! return a default position for our heroes
    static Point Standby() {
        return Point(Home().x() + sign * R_FOG_HERO, Home().y() + sign * R_FOG_HERO);
    }

private:
    bool is_done_{false};
    Point prev_{Home()};
    Point dest_{Home()};
};

class Opponent: public Entity {
public:
    Opponent(uint id, const Point& p): Entity(id, p, Type::kOpponent) {}

    friend std::ostream& operator<<(std::ostream& os, const Opponent& o) {
        os << "Opponent(" << o.id() << ", " << o.p().x() << ", " << o.p().y() << ")";
        return os;
    }
};

enum class Action
{
    kMove,
    kWind
};

class Order {
private:
    Action action_;
    Point dest_;

public:
    Order(): action_(Action::kMove), dest_(Hero::Standby()) {}
    Order(Action action, const Point& dest): action_(action), dest_(dest) {}

    Action action() const { return action_; }
    const Point& dest() const { return dest_; }

    friend ostream& operator<<(ostream& os, const Order& order) {
        switch (order.action()) {
            case Action::kMove:
                os << "MOVE ";
                break;
            case Action::kWind:
                os << "WIND ";
                break;
        }
        os << order.dest().x() << " " << order.dest().y();
        return os;
    }
};

//! Return the distance between 2 Points.
uint dist(const Point& a, const Point& b) {
    return static_cast<uint>(norm2(
        a.x() > b.x() ? a.x() - b.x() : b.x() - a.x(),
        a.y() > b.y() ? a.y() - b.y() : b.y() - a.y()));
}

//! Print to stderr to debug
template<typename... Args>
void debug(Args&&... args) {
    cerr << cat(args...) << endl;
}

class Solver {
public:
    int health;  // base health
    int mana;  // Spend ten mana to cast a spell
    map<int, Hero> heroes;
    map<int, Monster> monsters;
    deque<int> dangers, others;
    array<Order, 3> hero_orders{};

    void clear() {
        monsters.clear();
        dangers.clear();
    }

    uint get_closest_hero(const Monster& monster) const {
        double min_dist = numeric_limits<double>::max();
        int closest = NB_HEROES + 1;
        for (uint i = 0; i < NB_HEROES; i++) {
            double dist = monster.dist_to(heroes.at(i).p());
            if (dist < min_dist) {
                min_dist = dist;
                closest = i;
            }
        }
        return closest;
    }

    void compute_heroes_action() {
        // list monsters targeting our bases
        for (auto& [id, monster] : monsters) {
            if (monster.threat() == Threat::kBase) {
                dangers.push_back(id);
            } else {
                others.push_back(id);
            }
        }

        // sort monsters by distance to base
        // TODO(JG): priorize monster with target ?
        sort(dangers.begin(), dangers.end(), [this](int a, int b) -> bool {
            return Monster::cmp(monsters.at(a), monsters.at(b));
        });
        sort(others.begin(), others.end(), [this](int a, int b) -> bool {
            return Monster::cmp(monsters.at(a), monsters.at(b));
        });

        // push away monster if too close of the base
        if (monsters.at(dangers.front()).dist_to(Home()) < 2 * MSTR_MOV
            and mana > WIND_MANA) {
            uint closest = get_closest_hero(monsters.at(dangers.front()));
            hero_orders[closest] = Order(Action::kWind, Home());
            heroes.at(closest).done();
            monsters.erase(dangers.front());
            dangers.pop_front();
        }

        // action for heroes not pushing away a monster
        for (uint i = 0; i < NB_HEROES; i++) {
            if (heroes.at(i).is_done()) {
                continue;
            }
            if (not dangers.empty()) {
                hero_orders[i] = Order(Action::kMove, monsters.at(dangers.front()).p());
                heroes.at(i).done();
                monsters.erase(dangers.front());
                dangers.pop_front();
            } else {
                if (monsters.empty()) {
                    hero_orders[i] = Order(Action::kMove, heroes.at(i).Scout());
                    heroes.at(i).done();
                } else {
                    uint nb_target = heroes.at(i).get_nb_target(monsters);
                    if (mana > 4 * WIND_MANA && nb_target) {
                        hero_orders[i] = Order(Action::kWind, Opponent());
                    } else if (mana > 2 * WIND_MANA && nb_target > 1) {
                        hero_orders[i] = Order(Action::kWind, Opponent());
                    } else {
                        hero_orders[i] =
                            Order(Action::kMove, monsters.at(others.front()).p());
                        monsters.erase(others.front());
                        others.pop_front();
                    }
                    heroes.at(i).done();
                }
            }
        }

        for (uint i = 0; i < NB_HEROES; i++) {
            // Write an action using cout. DON'T FORGET THE "<< endl"
            // To debug: cerr << "Debug messages..." << endl;

            // In the first league: MOVE <x> <y> | WAIT; In later leagues: | SPELL
            // <spellParams>;
            cout << hero_orders.at(i) << endl;
        }
    }
};

int main() {
    int base_x;  // The corner of the map representing your base
    int base_y;
    cin >> base_x >> base_y;
    cin.ignore();
    int heroes_per_player;  // Always 3
    cin >> heroes_per_player;
    cin.ignore();

    top_left = base_x == 0 ? true : false;
    sign = top_left ? 1 : -1;

    Solver solver;

    // game loop
    while (1) {
        solver.clear();
        for (int i = 0; i < 2; i++) {
            cin >> solver.health >> solver.mana;
            cin.ignore();
        }
        int entity_count;  // Amount of heros and monsters you can see
        cin >> entity_count;
        cin.ignore();
        for (int i = 0; i < entity_count; i++) {
            int id;  // Unique identifier
            int type;  // 0=monster, 1=your hero, 2=opponent hero
            int x;  // Position of this entity
            int y;
            int shield_life;  // Ignore for this league; Count down until shield spell
                              // fades
            int is_controlled;  // Ignore for this league; Equals 1 when this entity is
                                // under a control spell
            int health;  // Remaining health of this monster
            int vx;  // Trajectory of this monster
            int vy;
            int has_target;  // 0=monster with no target yet, 1=monster targeting a base
            int threat;  // Given this monster's trajectory, is it a threat to
                         // 1=your base, 2=your opponent's base, 0=neither
            cin >> id >> type >> x >> y >> shield_life >> is_controlled >> health >> vx
                >> vy >> has_target >> threat;
            cin.ignore();
            if (type == 0) {  // monsters
                solver.monsters.emplace(pair<int, Monster>(
                    id, {id, x, y, health, vx, vy, has_target, threat}));
            } else if (type == 1) {  // heroes
                if (not solver.heroes.count(id)) {
                    solver.heroes.emplace(pair<int&, Hero>(id, {id, x, y}));
                } else {
                    solver.heroes.at(id).update(x, y);
                }
            }
        }
        solver.compute_heroes_action();
    }
}
