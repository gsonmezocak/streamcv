// Bu kod, sayfa yüklendiğinde çalışır
document.addEventListener("DOMContentLoaded", () => {
    
    // ===================================================
    // === 1. SIGN_UP.HTML (GİRİŞ/KAYIT) SAYFA MANTIĞI ===
    // ===================================================
    const loginForm = document.getElementById("login-form");
    const signupForm = document.getElementById("signup-form");
    const authToggle = document.querySelectorAll('input[name="auth-toggle"]');
    const cvUploadBox = document.getElementById("cv-upload-box"); // CV yükleme kutusunun ID'si
    const formTitle = document.getElementById("form-title");
    const formSubtitle = document.getElementById("form-subtitle");

    // Giriş/Kayıt arasında geçiş yapıldığında
    if (authToggle.length > 0) {
        // Başlangıçta "Sign Up" seçili (HTML'nizdeki gibi)
        loginForm.style.display = "none";
        signupForm.style.display = "block";
        cvUploadBox.style.display = "block";
        formTitle.textContent = "Create an Account"; // Başlığı güncelle
        formSubtitle.textContent = "Start your journey with us."; // Alt başlığı güncelle

        authToggle.forEach(toggle => {
            toggle.addEventListener("change", (e) => {
                if (e.target.value === "Log In") {
                    loginForm.style.display = "block";
                    signupForm.style.display = "none";
                    cvUploadBox.style.display = "none"; // Giriş yaparken CV yüklemeyi gizle
                    formTitle.textContent = "Welcome Back"; // Başlığı güncelle
                    formSubtitle.textContent = "Sign in to continue."; // Alt başlığı güncelle
                } else {
                    loginForm.style.display = "none";
                    signupForm.style.display = "block";
                    cvUploadBox.style.display = "block"; // Kayıt olurken göster
                    formTitle.textContent = "Create an Account"; // Başlığı güncelle
                    formSubtitle.textContent = "Start your journey with us."; // Alt başlığı güncelle
                }
            });
        });
    }

    // KAYIT OLMA (SIGN UP) FORMU GÖNDERİLDİĞİNDE
    if (signupForm) {
        signupForm.addEventListener("submit", async (e) => {
            e.preventDefault(); // Sayfanın yeniden yüklenmesini engelle
            const email = signupForm.querySelector('input[type="email"]').value;
            const password = signupForm.querySelector('input[type="password"]').value;
            
            // Not: CV yüklemeyi bu akıştan ayırıyoruz. Önce kayıt olmalı.
            // HTML'nizde CV yükleme kayıt formunun parçasıysa, buraya eklenmeli.
            // Şimdilik sadece auth yapıyoruz.
            
            const response = await fetch("/api/signup", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ email, password })
            });
            
            const result = await response.json();
            if (result.success) {
                alert("Kayıt başarılı! Şimdi giriş yapabilirsiniz.");
                // Otomatik olarak "Log In" sekmesine geçiş yap
                document.querySelector('input[value="Log In"]').click();
            } else {
                alert("Hata: " + result.error);
            }
        });
    }

    // GİRİŞ YAPMA (LOG IN) FORMU GÖNDERİLDİĞİNDE
    if (loginForm) {
        loginForm.addEventListener("submit", async (e) => {
            e.preventDefault();
            const email = loginForm.querySelector('input[type="email"]').value;
            const password = loginForm.querySelector('input[type="password"]').value;

            const response = await fetch("/api/login", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ email, password })
            });

            const result = await response.json();
            if (result.success) {
                // Giriş başarılı, /profile sayfasına yönlendir
                window.location.href = "/profile";
            } else {
                alert("Hata: " + result.error);
            }
        });
    }

    // ===================================================
    // === 2. CODE.HTML (PROFİL) SAYFA MANTIĞI          ===
    // ===================================================
    const profileForm = document.getElementById("profile-form");
    
    // Bu sayfa /profile sayfasıysa...
    if (profileForm) {
        
        // 1. Sayfa yüklenir yüklenmez mevcut profili çek ve formu doldur
        (async () => {
            const response = await fetch("/api/profile");
            if (!response.ok) {
                // Token süresi dolmuş vb.
                window.location.href = "/"; // Giriş sayfasına at
                return;
            }
            const profile = await response.json();
            document.getElementById("full-name").value = profile.full_name || "";
            document.getElementById("headline").value = profile.headline || "";
            document.getElementById("experience-skills").value = profile.cv_text || "";
        })();

        // 2. Profil formu gönderildiğinde (HTML'deki "Publish Profile" butonu)
        profileForm.addEventListener("submit", async (e) => {
            e.preventDefault();
            
            // Form verilerini 'FormData' olarak toplar (dosya yükleme için bu şart)
            const formData = new FormData(profileForm);
            
            // "Publish Profile" butonuna "Kaydediliyor..." yaz
            const publishButton = profileForm.querySelector('button[type="submit"]');
            publishButton.disabled = true;
            publishButton.textContent = "Kaydediliyor...";

            const response = await fetch("/api/profile", {
                method: "POST",
                body: formData // FormData'yı doğrudan gönder
            });

            const result = await response.json();
            if (result.success) {
                alert("Profil başarıyla kaydedildi!");
                publishButton.disabled = false;
                publishButton.textContent = "Publish Profile";
            } else {
                alert("Hata: " + result.error);
                publishButton.disabled = false;
                publishButton.textContent = "Publish Profile";
            }
        });

        // 3. (EKSİK PARÇA) Auto-Match Butonu
        // HTML'nizde "Auto-Match" için bir buton yok.
        // Örnek olarak "Generate Profile Summary" butonunu bu iş için kullanalım.
        const autoMatchButton = document.getElementById("auto-match-button");
        if (autoMatchButton) {
            autoMatchButton.addEventListener("click", async () => {
                
                // Sonuçları göstereceğimiz bir alan (HTML'nize eklemelisiniz)
                const resultsContainer = document.getElementById("match-results");
                if (!resultsContainer) {
                    alert("Sonuçları göstermek için 'match-results' ID'li bir div gerekli.");
                    return;
                }
                
                resultsContainer.innerHTML = "<p>Eşleşmeler aranıyor... Bu işlem 30 saniye sürebilir.</p>";

                const response = await fetch("/api/auto-match");
                const results = await response.json();

                if (response.ok) {
                    resultsContainer.innerHTML = "<h3>En İyi Eşleşmeler</h3>";
                    if (results.length === 0) {
                        resultsContainer.innerHTML += "<p>Uygun bir eşleşme bulunamadı.</p>";
                        return;
                    }
                    // Sonuçları HTML olarak formatla
                    results.forEach((r, index) => {
                        resultsContainer.innerHTML += `
                            <div style="border: 1px solid #ddd; padding: 10px; margin-bottom: 10px; border-radius: 8px;">
                                <h4>#${index + 1}: ${r.job.title} (Skor: ${r.score}%)</h4>
                                <p><b>Özet:</b> ${r.data.summary}</p>
                                <p><b>Artılar:</b> ${r.data.pros.join(", ")}</p>
                                <p><b>Eksiler:</b> ${r.data.cons.join(", ")}</p>
                            </div>
                        `;
                    });
                } else {
                    resultsContainer.innerHTML = `<p style="color: red;">Hata: ${results.error}</p>`;
                }
            });
        }
    }
});
