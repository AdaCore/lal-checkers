procedure Test is
   X : aliased Integer;
   Y : access all Integer := X'Access;
begin
   Y.all := 12;
exception
   when others =>
      null;
end Test;
