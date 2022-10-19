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

template<typename C, typename D, typename T = typename C::value_type>
inline std::string JoinImpl(const C& container, const D& delimiter) {
    size_t n = 0UL;
    std::ostringstream oss;
    for (auto& el : container) {
        if (n == 0UL) {
            oss << el;
        } else {
            oss << delimiter << el;
        }
        n++;
    }
    return oss.str();
}
}  // end namespace detail_string

//! Concatenation of y_ streamable objects into a string, returns a std::string
template<typename... Args>
inline std::string cat(Args&&... args) {
    std::ostringstream oss;
    detail_string::PushToStream(oss, std::forward<Args>(args)...);
    return oss.str();
}

//! Simple join of containers into a string, with a char delimiter
template<
    typename C,
    typename T = typename C::value_type,
    typename = typename std::enable_if<
        detail_string::is_streamable<std::ostringstream, T>{}>::type>
inline std::string join(C const& container, char delimiter = ',') {
    return detail_string::JoinImpl(container, delimiter);
}

//! Simple join of containers into a string, with a string delimiter
template<
    typename C,
    typename T = typename C::value_type,
    typename = typename std::enable_if<
        detail_string::is_streamable<std::ostringstream, T>{}>::type>
inline std::string join(C const& container, const std::string& delimiter = ", ") {
    return detail_string::JoinImpl(container, delimiter);
}

//! Simple splitting of a string, over a char
inline std::vector<std::string> split(const std::string& s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream iss(s);
    while (std::getline(iss, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

//! Simple splitting of a string, over a string
inline std::vector<std::string> split(
    const std::string& s, const std::string& delimiter) {
    std::vector<std::string> tokens;
    size_t pos_start = 0UL;
    size_t pos_end = s.find(delimiter);
    size_t delim_len = delimiter.length();

    while (pos_end != std::string::npos) {
        tokens.push_back(s.substr(pos_start, pos_end - pos_start));
        pos_start = pos_end + delim_len;
        pos_end = s.find(delimiter, pos_start);
    }
    tokens.push_back(s.substr(pos_start));
    return tokens;
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

//! Print to stderr to debug
template<typename... Args>
void debug(Args&&... args) {
    cerr << cat(args...) << endl;
}

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
constexpr uint SHIELD_TIME{12u};  // wind spell movement
constexpr uint FOCUS_RANGE{5000u};
constexpr uint R_SHIELD{2200u};
constexpr uint R_CTRL{2200u};
constexpr uint MANA_COST{10u};
constexpr uint MSTR_SHIELD_MOV = SHIELD_TIME * MSTR_MOV;

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
        x_ = min(x_, X_MAX);
        y_ = min(y_, Y_MAX);
    }
    void advance(const Vector2& speed) {
        x_ += speed.x();
        y_ += speed.y();
        x_ = min(x_, X_MAX);
        y_ = min(y_, Y_MAX);
    }
    void rotate(double phi) {
        const double sinphi = sin(phi);
        const double cosphi = cos(phi);
        double xd = static_cast<double>(x_);
        double yd = static_cast<double>(y_);

        const double t = xd * cosphi - yd * sinphi;
        yd = yd * cosphi + xd * sinphi;

        xd = min(max(t, static_cast<double>(X_MIN)), static_cast<double>(X_MAX));
        yd = min(max(yd, static_cast<double>(Y_MIN)), static_cast<double>(Y_MAX));

        x_ = static_cast<uint>(xd);
        y_ = static_cast<uint>(yd);
    }

    friend std::ostream& operator<<(std::ostream& os, const Point& p) {
        os << "Point(" << p.x() << ", " << p.y() << ")";
        return os;
    }

    bool operator==(const Point& other) const { return x_ == other.x() and y_ == other.y(); }
    Point operator+(const Point& other) const {
        return Point(x_ + other.x(), y_ + other.y());
    }
    Point operator+(const Vector2& speed) const {
        return Point(x_ + speed.x(), y_ + speed.y());
    }
    Point operator-(const Point& other) const {
        return Point(x_ - other.x(), y_ - other.y());
    }
    Point operator-(const Vector2& speed) const {
        return Point(x_ - speed.x(), y_ - speed.y());
    }
    Point operator*(uint d) { return Point(d * x_, d * y_); }
};

enum class Type
{
    kMonster,
    kHero,
    kVilain
};

// global variable to indicate if our base is top left (else bottom rifght)
static const array<Point, 2> kBases{Point{X_MAX, Y_MAX}, Point{X_MIN, Y_MIN}};
static const Point kCorner(X_MAX, Y_MIN);
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
    Entity(uint id, const Point& p, Type type, int shield, int is_controlled):
        id_(id), p_(p), type_(type), shield_(shield), is_controlled_(is_controlled) {}
    virtual ~Entity() {}
    Entity(const Entity&) = default;
    constexpr Entity(Entity&&) = default;
    Entity& operator=(const Entity&) = default;
    Entity& operator=(Entity&&) = default;

    uint id() const { return id_; }
    Point p() const { return p_; }
    Type type() const { return type_; }
    uint shield() const { return shield_; }
    bool is_controlled() const { return is_controlled_; }

    double dist_to(const Point& p) const { return p_.dist_to(p); }

    void update(uint x, uint y, int shield, int is_controlled) {
        p_ = Point(x, y);
        shield_ = shield;
        is_controlled_ = is_controlled;
    }

    friend std::ostream& operator<<(std::ostream& os, const Entity& e) {
        os << "Entity(" << e.id() << ", " << e.p().x() << ", " << e.p().y() << ")";
        return os;
    }

private:
    uint id_;
    Point p_;
    Type type_;
    uint shield_;
    bool is_controlled_;
};

enum class Threat
{
    kNone,
    kBase,
    kOpp
};

Point threat_destination(Threat threat) {
    switch(threat) {
    case Threat::kBase: 
        return Home();
    case Threat::kOpp: 
        return Opponent();
    default:
        return Point();
    }
}

class Monster: public Entity {
public:
    Monster(
        uint id,
        const Point& p,
        int shield,
        int is_controlled,

        uint health,
        int vx,
        int vy,
        bool has_target,
        int threat):
        Entity(id, p, Type::kMonster, shield, is_controlled),
        health_(health),
        v_(vx, vy),
        has_target_(has_target),
        threat_(Threat(threat)) {}
    Monster(
        int id,
        int x,
        int y,
        int shield,
        int is_controlled,
        int health,
        int vx,
        int vy,
        int has_target,
        int threat):
        Entity(
            static_cast<uint>(id),
            {static_cast<uint>(x), static_cast<uint>(y)},
            Type::kMonster,
            shield,
            is_controlled),
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
        uint x,
        uint y,
        int shield,
        int is_controlled,
        uint health,
        int vx,
        int vy,
        bool has_target,
        int threat) {
        Entity::update(x, y, shield, is_controlled);
        health_ = health;
        v_ = Vector2(vx, vy);
        has_target_ = has_target;
        threat_ = Threat(threat);
    }

    int turn_to_kill() {
        if (not has_target()) {
            numeric_limits<int>::max();
        }
        return static_cast<int>(ceil((p().dist_to(threat_destination(threat_)) - R_BASE) / MSTR_MOV));
    }

    bool soloable() {
        return turn_to_kill() * DMG >= health_; 
    }

    friend std::ostream& operator<<(std::ostream& os, const Monster& m) {
        os << "Monster(" << m.id() << ", " << m.p().x() << ", " << m.p().y() << ", "
           << m.health() << ", " << m.v() << ", "
           << (m.has_target() ? "targetting" : "wandering") << ", " << m.threat()
           << ")";
        return os;
    }

    static bool cmp(const Monster& a, const Monster& b) {
        auto d_a = a.dist_to(Home());
        auto d_b = b.dist_to(Home());
        return d_a < d_b;
    }

    static bool anti_cmp(const Monster& a, const Monster& b) {
        auto d_a = a.dist_to(Opponent());
        auto d_b = b.dist_to(Opponent());
        return d_a > d_b;
    }

private:
    uint health_;
    Vector2 v_;
    bool has_target_;
    Threat threat_;
};

class Hero: public Entity {
public:
    Hero(uint id, const Point& p, int shield, int is_controlled):
        Entity(id, p, Type::kHero, shield, is_controlled) {}
    Hero(int id, int x, int y, int shield, int is_controlled):
        Entity(
            static_cast<uint>(id),
            {static_cast<uint>(x), static_cast<uint>(y)},
            Type::kHero,
            shield,
            is_controlled) {}

    void update(uint x, uint y, int shield, int is_controlled) {
        prev_ = p();
        Entity::update(x, y, shield, is_controlled);
        is_done_ = false;
        has_been_controlled_ |= is_controlled;
    }

    void done() { is_done_ = true; }
    bool is_done() const { return is_done_; }
    bool has_been_controlled() const { return has_been_controlled_; }

    Point scout() {
        if (id() == 0) {
            return standby();
        }

        if (id() == 1) {
            Point dest;
            double dist = dist_to(Home());
            if (dist > 2u * R_FOG_BASE) {
                dest = Home();
            } else if (dist > R_FOG_BASE) {
                dest = dest_;
            } else {
                dest = Opponent();
            }
            dest_ = dest;
            return dest;
        }

        Point dest;
        double dist = dist_to(Opponent());
        if (dist > R_FOG_BASE + R_FOG_BASE / 2u) {
            dest = Opponent();
        } else if (dist > R_FOG_BASE / 4u * 3u) {
            dest = dest_;
        } else {
            dest = Home();
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
    static Point standby() {
        return Point(Home().x() + sign * R_FOG_HERO, Home().y() + sign * R_FOG_HERO);
    }

private:
    bool is_done_{false};
    bool has_been_controlled_{false};
    Point prev_{Home()};
    Point dest_{Home()};
};

class Vilain: public Entity {
public:
    Vilain(uint id, const Point& p, int shield, int is_controlled):
        Entity(id, p, Type::kVilain, shield, is_controlled) {}
    Vilain(int id, int x, int y, int shield, int is_controlled):
        Entity(
            static_cast<uint>(id),
            {static_cast<uint>(x), static_cast<uint>(y)},
            Type::kVilain,
            shield,
            is_controlled) {}

    void update(uint x, uint y, int shield, int is_controlled) {
        Entity::update(x, y, shield, is_controlled);
    }

    friend std::ostream& operator<<(std::ostream& os, const Vilain& o) {
        os << "Vilain(" << o.id() << ", " << o.p().x() << ", " << o.p().y() << ")";
        return os;
    }
};

enum class Action
{
    kMove,
    kWind,
    kShield,
    kControl
};

class Order {
private:
    Action action_;
    Point dest_;
    uint id_;

public:
    Order(): action_(Action::kMove), dest_(Hero::standby()), id_{0u} {}
    Order(Action action, const Point& dest, uint id = 0u):
        action_(action), dest_(dest), id_(id) {
        debug(*this);
    }

    Action action() const { return action_; }
    const Point& dest() const { return dest_; }

    friend ostream& operator<<(ostream& os, const Order& order) {
        switch (order.action()) {
            case Action::kMove:
                os << "MOVE " << order.dest().x() << " " << order.dest().y();
                break;
            case Action::kWind:
                os << "SPELL WIND " << order.dest().x() << " " << order.dest().y();
                break;
            case Action::kShield:
                os << "SPELL SHIELD " << order.id_;
                break;
            case Action::kControl:
                os << "SPELL CONTROL " << order.id_ << " " << order.dest().x() << " "
                   << order.dest().y();
                break;
        }
        return os;
    }
};

//! Return the distance between 2 Points.
uint dist(const Point& a, const Point& b) {
    return static_cast<uint>(norm2(
        a.x() > b.x() ? a.x() - b.x() : b.x() - a.x(),
        a.y() > b.y() ? a.y() - b.y() : b.y() - a.y()));
}

class Solver {
public:
    int health;  // base health
    int mana;  // Spend ten mana to cast a spell
    map<int, Hero> heroes;
    map<int, Vilain> vilains;
    map<int, Monster> monsters;
    deque<int> dangers, others, friends;
    array<Order, 3> hero_orders{};

    void clear() {
        monsters.clear();
        dangers.clear();
        others.clear();
        friends.clear();
        vilains.clear();
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

    pair<int, double> get_closest_monster_to_opponent() const {
        double min_dist = numeric_limits<double>::max();
        int closest = -1;
        for (const auto& [id, monster] : monsters) {
            double dist = monster.dist_to(Opponent());
            if (dist < min_dist) {
                min_dist = dist;
                closest = id;
            }
        }
        return {closest, min_dist};
    }

    pair<int, double> get_closest_vilain(const Hero& hero) const {
        double min_dist = numeric_limits<double>::max();
        int closest = -1;
        for (const auto& [id, vilain] : vilains) {
            double dist = vilain.dist_to(hero.p());
            if (dist < min_dist) {
                min_dist = dist;
                closest = id;
            }
        }
        return {closest, min_dist};
    }

    void compute_heroes_action() {
        debug("compute_heroes_action");
        debug("monsters: ", monsters.size());
        // list monsters targeting our bases
        for (const auto& [id, monster] : monsters) {
            if (monster.threat() == Threat::kBase) {
                dangers.push_back(id);
            } else if (monster.threat() == Threat::kOpp) {
                friends.push_back(id);
            } else {
                others.push_back(id);
            }
        }

        // sort monsters by distance to base
        // TODO(JG): priorize monster with target ?
        debug("dangers: ", dangers.size(), "=>", join(dangers, ", "));
        if (dangers.size() > 1) {
            sort(dangers.begin(), dangers.end(), [this](int a, int b) -> bool {
                return Monster::cmp(monsters.at(a), monsters.at(b));
            });
        }
        debug("others: ", others.size(), "=>", join(others, ", "));
        if (others.size() > 1) {
            sort(others.begin(), others.end(), [this](int a, int b) -> bool {
                return Monster::cmp(monsters.at(a), monsters.at(b));
            });
        }
        debug("friends: ", friends.size(), "=>", join(friends, ", "));
        if (friends.size() > 1) {
            sort(friends.begin(), friends.end(), [this](int a, int b) -> bool {
                return Monster::cmp(monsters.at(a), monsters.at(b));
            });
        }

        if (not dangers.empty()) {
            debug(
                "push monster: dist=",
                monsters.at(dangers.front()).dist_to(Home()),
                " <? ",
                2 * MSTR_MOV,
                ", mana=",
                mana,
                " >? ",
                MANA_COST);
        }
        try {
            // push away unshield monster if too close of the baseCorner
            if (not dangers.empty()
                and monsters.at(dangers.front()).dist_to(Home()) < 2 * MSTR_MOV
                and mana > MANA_COST
                and not monsters.at(dangers.front()).shield())
            {
                uint closest = get_closest_hero(monsters.at(dangers.front()));
                debug("Order: ", closest, " => wind to opp");
                hero_orders[closest] = Order(Action::kWind, Opponent());
                heroes.at(closest).done();
                monsters.erase(dangers.front());
                dangers.pop_front();
            }
        } catch (const exception& xcpt) {
            debug("push monster: xcpt.what(), ", dangers.front());
            debug("closest: ", get_closest_hero(monsters.at(dangers.front())));
        }

        // action for heroes not pushing away a monster
        for (uint i = 0; i < NB_HEROES - 1; i++) {
            if (heroes.at(i).is_done()) {
                continue;
            }

            for (int id : dangers) {
                auto vec = Vector2::from_polar(R_WIND, (Opponent() - monsters.at(id).p()).theta());
                Point landing = monsters.at(id).p() + vec;
                debug("landing ", id, " ", landing);
                if (landing.dist_to(Home()) > FOCUS_RANGE) {
                    debug("landing OUTSIDE");
                }
            }

            if (false) {

            } else if (auto [id, d] = get_closest_vilain(heroes.at(i)); id != -1
                and vilains.at(id).p().dist_to(Home()) <= FOCUS_RANGE
                and not vilains.at(id).shield()
                and d <= R_CTRL
                and mana > 3 * MANA_COST)
            {          
                debug("Order: ", i, " => ctrl def: ", id);
                hero_orders[i] = Order(Action::kControl, Opponent(), id);
                heroes.at(i).done();
            } else if (not dangers.empty()) {
                debug("Order: ", i, " => attack: ", monsters.at(dangers.front()).p());
                hero_orders[i] = Order(Action::kMove, monsters.at(dangers.front()).p());
                heroes.at(i).done();
                // only remove from dangers if not a big threat
                if (monsters.at(dangers.front()).soloable()) {
                    monsters.erase(dangers.front());
                    dangers.pop_front();
                }
            } else {
                debug("no danger");
                if (heroes.at(0).has_been_controlled() and not heroes.at(0).shield()) {
                    debug("Order: ", i, " => preemptive protec: ", heroes.at(0).id());
                    hero_orders[i] = Order(Action::kShield, Point(), heroes.at(0).id());
                    heroes.at(i).done();
                } else if (others.empty()) {
                    debug("Order: ", i, " => scout: ", heroes.at(i).scout());
                    hero_orders[i] = Order(Action::kMove, heroes.at(i).scout());
                    heroes.at(i).done();
                } else {
                    uint nb_target = heroes.at(i).get_nb_target(monsters);
                    if (mana > 3 * MANA_COST and nb_target > 1) {
                        debug("Order: ", i, " => wind to opp");
                        hero_orders[i] = Order(Action::kWind, Opponent());
                    } else {
                        debug(
                            "Order: ", i, " => atck:", monsters.at(others.front()).p());
                        hero_orders[i] =
                            Order(Action::kMove, monsters.at(others.front()).p());
                        monsters.erase(others.front());
                        others.pop_front();
                    }
                    heroes.at(i).done();
                }
            }
        }

        if (not heroes.at(2).is_done()) {
            Hero& h = heroes.at(2);

            if (not h.is_done()) {
                uint nb_target = h.get_nb_target(monsters);
                if (mana > 3 * MANA_COST and nb_target > 2) {
                    debug("Order: 2 => wind to opp");
                    hero_orders[2] = Order(Action::kWind, Opponent());
                    h.done();
                }
            }

            if (not h.is_done()) {
                for (const auto& [id, monster] : monsters) {
                    if (mana > 3 * MANA_COST
                        and monster.dist_to(Opponent()) < MSTR_SHIELD_MOV
                        and monster.dist_to(h.p()) < R_SHIELD and not monster.shield()
                        and not monster.is_controlled())
                    {
                        debug("Order: 2 => Shield mstr: ", monster);
                        hero_orders[2] = Order(Action::kShield, Home(), id);
                        h.done();
                    }
                }
            }

            if (not h.is_done()) {
                for (const auto& [id, vilain] : vilains) {
                    if (mana > 3 * MANA_COST and vilain.dist_to(h.p()) < R_CTRL
                        and vilain.dist_to(Opponent()) < R_FOG_BASE
                        and not vilain.shield() and not vilain.is_controlled())
                    {
                        debug("Order: 2 => Control vilain: ", vilain);
                        hero_orders[2] = Order(Action::kControl, Home(), id);
                        h.done();
                    }
                }
            }

            if (not h.is_done()) {
                for (const auto& [id, monster] : monsters) {
                    if (mana > 3 * MANA_COST and monster.dist_to(h.p()) < R_CTRL
                        and monster.threat() != Threat::kOpp and not monster.shield()
                        and not monster.is_controlled())
                    {
                        debug("Order: 2 => Control monster: ", monster);
                        hero_orders[2] = Order(Action::kControl, Opponent(), id);
                        h.done();
                    }
                }
            }

            //             if (not h.is_done() and not others.empty()) {
            //                 double min_dist = numeric_limits<double>::max();
            //                 int closest = -1;
            //                 for (int other : others) {
            //                     double d = h.dist_to(monsters.at(other).p());
            //                     if (d < min_dist) {
            //                         min_dist = d;
            //                         closest = other;
            //                     }
            //                 }
            //                 debug("Order: 2 => attack: ", closest);
            //                 hero_orders[2] = Order(Action::kMove,
            //                 monsters.at(closest).p()); h.done();
            //             }

            if (not h.is_done()) {
                debug("Order: 2 => scout: ", h.scout());
                hero_orders[2] = Order(Action::kMove, h.scout());
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
    int dump;
    while (1) {
        solver.clear();
        cin >> solver.health >> solver.mana;
        cin.ignore();
        cin >> dump >> dump;
        cin.ignore();
        int entity_count;  // Amount of heros and monsters you can see
        cin >> entity_count;
        cin.ignore();
        for (int i = 0; i < entity_count; i++) {
            int id;  // Unique identifier
            int type;  // 0=monster, 1=your hero, 2=opponent hero
            int x;  // Position of this entity
            int y;
            int shield;  // Count down until shield spell fades
            int is_controlled;  // Equals 1 when this entity is under a control spell
            int health;  // Remaining health of this monster
            int vx;  // Trajectory of this monster
            int vy;
            int has_target;  // 0=monster with no target yet, 1=monster targeting a base
            int threat;  // Given this monster's trajectory, is it a threat to
                         // 1=your base, 2=your opponent's base, 0=neither
            cin >> id >> type >> x >> y >> shield >> is_controlled >> health >> vx >> vy
                >> has_target >> threat;
            cin.ignore();
            if (type == 0) {  // monsters
                solver.monsters.emplace(pair<int, Monster>(
                    id,
                    {id,
                     x,
                     y,
                     shield,
                     is_controlled,
                     health,
                     vx,
                     vy,
                     has_target,
                     threat}));
            } else if (type == 1) {  // heroes
                if (not top_left) {
                    id -= 3;
                }
                if (not solver.heroes.count(id)) {
                    solver.heroes.emplace(
                        pair<int&, Hero>(id, {id, x, y, shield, is_controlled}));
                } else {
                    solver.heroes.at(id).update(x, y, shield, is_controlled);
                }
            } else {
                solver.vilains.emplace(
                    pair<int&, Vilain>(id, {id, x, y, shield, is_controlled}));
            }
        }
        solver.compute_heroes_action();
    }
}
