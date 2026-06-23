self.addEventListener("push", function (event) {
    let data = {};

    try {
        data = event.data ? event.data.json() : {};
    } catch (error) {
        data = {
            title: "TNEB Smart Alert",
            body: event.data ? event.data.text() : "You have a new alert."
        };
    }

    const title = data.title || "TNEB Smart Alert";

    const options = {
        body: data.body || data.message || "You have a new electricity usage alert.",
        icon: data.icon || "/static/icon-192.png",
        badge: data.badge || "/static/icon-192.png",
        data: {
            url: data.url || "/"
        },
        vibrate: [100, 50, 100],
        tag: data.tag || "tneb-smart-alert",
        renotify: true
    };

    event.waitUntil(
        self.registration.showNotification(title, options)
    );
});

self.addEventListener("notificationclick", function (event) {
    event.notification.close();

    const url = (event.notification.data && event.notification.data.url) || "/";

    event.waitUntil(
        clients.matchAll({
            type: "window",
            includeUncontrolled: true
        }).then(function (clientList) {
            for (const client of clientList) {
                if ("focus" in client) {
                    client.navigate(url);
                    return client.focus();
                }
            }

            if (clients.openWindow) {
                return clients.openWindow(url);
            }
        })
    );
});